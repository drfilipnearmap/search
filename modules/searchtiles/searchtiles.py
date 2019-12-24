import glob, os, pathlib, matplotlib, scipy, PIL, random, requests, json, math, hnswlib, pygis
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, rotate
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pyimage_ops.tiler import TileRequester
from pyimage_ops.pred_tiler.requester import PredictionRequester
from skimage.draw import polygon
from skimage.measure import find_contours, approximate_polygon
from spoor.smoothing_utils import get_hough_transform_angle
from training.dilated_resnet_fcn import make_dilated_fcn_resnet_16s


def fetch_tiles(lat, lon, zoom_level, datestr, box_size, file_path, categories = [-1], verbose = False):
    
    """
    Use the prediction API (through pyimage_ops) to grab a large square of prediction raster tiles
    
    Arguments:
    lat (float): Latitude of the centre of the box.
    lon (float): Longitude of the centre of the box. 
    zoom_level (int): Zoom level at which to grab tiles.
    datestr (string): Survey date to use for the tiles.
    box_size (int): Edge size of the square which is to be fetched, where the units are tiles. eg box_size = 600 will grab a 600x600 square chunk centred around (lon,lat) 
    file_path (string): Directory in which to save the resulting images.
    categories (list of ints): Optional list to limit the prediction categories fetched. If this is [-1], which is the default, all categories will be fetched.
    verbose (boolean): Whether or not to print tqdm progress bars while running.
    
    Returns:
    None. All fetched images are saved to the specified directory. 
    """
    
    if file_path[-1] == '/':
        file_path = file_path[:-1]
    
    total_box = int(box_size/2)
    
    # Fetch information about which predictions exist using the coverage API
    coord = pygis.conversions.GeoCoordinates()
    tx_start, ty_start = coord.long_lat_to_tile_coordinate(lon, lat, zoom_level)
    url = ("https://api.nearmap.com/coverage/v2/coord/{}/{}/{}?apikey={}&until={}&limit=1"
           .format(zoom_level, tx_start, ty_start, os.environ["API_KEY"], datestr))
    response = requests.get(url).json()
    survey = response["surveys"][0]["id"]
    
    # By default, just grab all the classes available from the API
    if categories[0] != -1:
        classes = categories
    else:
        classes = response["surveys"][0]["resources"]["predictions"][0]["properties"]["classes"]
        classes = [int(r['id']) for r in classes]
    
    
    pred_req = PredictionRequester()
    
    # The bounding box is iterated through in a rolling window of size (2*dx + 1), (2*dy + 1)
    # I don't recommend changing this. If you do, keep them equal to each other.
    dx = 2
    dy = 2

    # Construct the required folder structure so you don't have to do it in the loop
    os.system('mkdir -p {}'.format(file_path))
    for cat in classes:
        os.system('mkdir {}/{}/'.format(file_path, cat))

    window_size = 2*dx+1

    # Iterate through the bounding box row by row in windows 
    for tx in tqdm_notebook(range(tx_start-total_box, tx_start+total_box, window_size), desc='Outer loop (x)', disable = not verbose):
        for ty in tqdm_notebook(range(ty_start-total_box, ty_start+total_box, window_size), desc='Inner loop (y)', disable = not verbose):

            # Grab a window of tiles. It's faster to use the API to grab multiple tiles stitched at once,
            # but grabbing too many will result in running out of memory
            tiles = pred_req.get_preds_bb_tc_with_survey(zoom_level, tx-dx, ty-dy, tx+dx, ty+dy, survey, classes)
            tiles_split = np.split(tiles, 2*dy+1)

            # Iterate through each window, saving every individual tile
            for row in range(0, len(tiles_split)):
                tiles_split[row] = np.split(tiles_split[row], 2*dx+1, 1)
                for col in range(len(tiles_split[row])):
                    x = tx-dx+col
                    y = ty-dy+row
                    tiles_split[row][col] = np.array(tiles_split[row][col])

                    # Save each prediction category in a separate folder so they can be quickly grabbed later
                    for i, cat in enumerate(classes):
                        im = PIL.Image.fromarray((tiles_split[row][col][:,:,i]*255).astype(np.uint8), mode = "L")
                        im.save('{}/{}/{}_{}.jpeg'.format(file_path, cat, x, y))

def load_predictions(file_path, category = 1, cull = 1, size = 128, verbose = True, get_predictions = True, get_coordinates = True, start = 0, total = -1):
    
    """
    Load prediction tiles for a given category from the disk. Tiles must be stored in a folder corresponding to their category and be named x_y<_cropindex>.jpeg. The crop index is optional, and will only pop up if the tiles were generated with the crop_tiles() function. For example, a folder 'sydney_tiles_2019' may have three subfolders '1', '5' and '15' for those three categories. Each of the subfolder contains tiles named '7438_192.jpeg', '7438_193.jpeg', etc.

    Arguments:
    file_path (string): Directory of the prediction tiles.
    category (int): Prediction category from https://confluence.nearmap.com/x/dAzsAg.
    cull (int): Take one prediction for every <cull> files. For example if cull = 10, only 1/10 of the files will be loaded.
    size (int): Desired tile size - each tile will be resized to [size, size].
    verbose (boolean): Whether or not to print extra information while the function is running.
    get_predictions and get_coordinates (boolean): whether or not to return predictions or coordinates. predictions take a long time, and sometimes you just need coordinates only.
    start (int): Index at which to start loading images from the folder.
    total (int): Total amount of images to load. -1 means load all of them, starting from 'start'.

    Returns:
    (predictions, coordinates) Tuple.
    predictions is [num_files, tile_size_x, tile_size] dimensions, storing all loaded tiles.
    coordinates is [num_files, 2] dimensions, storing x and y tile coordinates respectively for each loaded tile.
    If the tiles were generated through cropping they'll have an additional index, so coordinates will be [num_files, 3] where the extra variable is the index.
    """
    
    if file_path[-1] == '/':
        file_path = file_path[:-1]

    predictions = []
    coordinates = []
    i = 1
    owd = os.getcwd()
    os.chdir("{}/{}/".format(file_path, category))
    
    if total == -1:
        total = len(glob.glob('*.jpeg'))
    
    # For each tile in the folder, load its contents in to predictions, scale it down if necessary and strip its file name for its coordinates
    
    file_list_full = glob.glob("*.jpeg")
    file_list_full.sort()
    file_list = file_list_full[start:start+total].copy()
    
    for file in tqdm_notebook(file_list, disable = not verbose, desc = "Loading predictions"):
        if i%cull == 0:
            if get_predictions == True:
                im = np.array(PIL.Image.open(file)).astype(np.uint8)
                scaling = size/im.shape[0]
                im_small = zoom(im, [scaling, scaling], order=0)
                predictions.append(im_small)
            
            if get_coordinates == True:
                file_name = str(file).strip('.jpeg').split('_')
                x = int(file_name[0])
                y = int(file_name[1])
                if len(file_name) > 3:
                    split = int(file_name[2])
                    index = int(file_name[3])
                    coordinates.append([x, y, split, index])
                else:
                    coordinates.append([x, y])
            i = 0
        i += 1
    
    os.chdir(owd)
    
    if get_predictions == True and get_coordinates == True:
        return (np.array(predictions).astype(np.float32), np.array(coordinates).astype(np.int32))
    
    if get_predictions == True and get_coordinates == False:
        return np.array(predictions).astype(np.float32)
    
    if get_predictions == False and get_coordinates == True:
        return np.array(coordinates).astype(np.int32)

def load_predictions_from_file_list(file_list, cull = 1, size = 128, verbose = True, get_predictions = True, get_coordinates = True):
    """
    Same as above load_predictions() but instead of providing a folder, a list of files is provided. This is useful when only grabbing a specific section, or multiple folders.

    Load prediction tiles for a given directoryfrom the disk. The crop index is optional, and will only pop up if the tiles were generated with the crop_tiles() function. For example, a folder 'sydney_tiles_2019' may have three subfolders '1', '5' and '15' for those three categories. Each of the subfolder contains tiles named '7438_192.jpeg', '7438_193.jpeg', etc.

    Arguments:
    file_path (string): Directory of the prediction tiles.
    cull (int): Take one prediction for every <cull> files. For example if cull = 10, only 1/10 of the files will be loaded.
    size (int): Desired tile size - each tile will be resized to [size, size].
    verbose (boolean): Whether or not to print extra information while the function is running.
    get_predictions and get_coordinates (boolean): whether or not to return predictions or coordinates. predictions take a long time, and sometimes you just need coordinates only.
    start (int): Index at which to start loading images from the folder.
    total (int): Total amount of images to load. -1 means load all of them, starting from 'start'.

    Returns:
    (predictions, coordinates) tuple, or just predictions or coordinates depending on the value of get_predictions and get_coordinates.
    predictions is [num_files, tile_size_x, tile_size] dimensions, storing all loaded tiles.
    coordinates is [num_files, 2] dimensions, storing x and y tile coordinates respectively for each loaded tile. 
    If the tiles were generated through cropping they'll have an additional index, so coordinates will be [num_files, 3] where the extra variable is the index.
    """
    predictions = []
    coordinates = []
    i = 1
    
    
    # For each tile in the folder, load its contents in to predictions, scale it down if necessary and strip its file name for its coordinates
    for file in tqdm_notebook(file_list, disable = not verbose, desc = "Loading predictions"):
        if i%cull == 0:
            
            if get_predictions == True:
                im = np.array(PIL.Image.open(file)).astype(np.uint8)
                scaling = size/im.shape[0]
                im_small = zoom(im, [scaling, scaling], order=0)
                predictions.append(im_small)
            
            if get_coordinates == True:
                image_name = str(file).split('/')[-1]
                file_name = str(image_name).strip('.jpeg').split('_')
                x = int(file_name[0])
                y = int(file_name[1])
                if len(file_name) > 3:
                    split = int(file_name[2])
                    index = int(file_name[3])
                    coordinates.append([x, y, split, index])
                else:
                    coordinates.append([x, y])
            i = 0
        i += 1
    
    if get_predictions == True and get_coordinates == True:
        return (np.array(predictions).astype(np.float32), np.array(coordinates).astype(np.float32))
    
    if get_predictions == True and get_coordinates == False:
        return np.array(predictions).astype(np.float32)
    
    if get_predictions == False and get_coordinates == True:
        return np.array(coordinates).astype(np.float32)

    
    
def crop_training_predictions(images, zoom_level):
    
    """
    Take a set of large Apollo images and crop it in to smaller images of a given zoom level. This function shouldn't be manually called, it's used by dataset_from_apollo(). Currently only uses 896x896 images, giving 1 z20, 9 z21 and 49 z22 images.

    Arguments:
    images (list of NxNx3 numpy arrays): A list of images from Apollo's dataset, which are 896x896x3 or greater.

    Returns:
    A list of 128x128 numpy arrays, representing the cropped images.
    """
    
    if zoom_level not in [19, 20, 21, 22]:
        print('Invalid zoom level, please use 19, 20, 21 or 22 only')
        return None
    
    img_list = []
    
    for i in images:
        
        # 896x896 images are not z19, but there are some 1152x1152 and 1408x1408 images that work fine. These were cropped to 1024x1024 in dataset_from_apollo()
        if zoom_level == 19 and i.shape[0] == 1024:
            img_list.append(zoom(i, (0.125, 0.125)))
            
        # Just take the top left corner for zoom 20 stuff. 
        # Most of the time we get 896x896 which means we can't have multiple z20 crops in one
        if zoom_level == 20 and i.shape[0] == 896:
            stitched = i[0:512, 0:512]
            img_list.append(zoom(stitched, (0.25, 0.25)))
        
        if zoom_level == 21 and i.shape[0] == 896:
            for row in range(3):
                for col in range(3):
                    stitched = i[256*row:256*(row+1), 256*col:256*(col+1)]
                    img_list.append(zoom(stitched, (0.5, 0.5)))

        elif zoom_level == 22 and i.shape[0] == 896:
            for row in range(7):
                for col in range(7):
                    img_list.append(i[row*896//7:(row+1)*896//7, 
                                        col*896//7:(col+1)*896//7])
    
    return img_list


def dataset_from_apollo(training_paths, test_paths, zoom_level=21):
    
    """
    Grabs all 896x896 apollo training images from the given paths. This is used in batch_inference(), and shouldn't be called manually.
    
    Arguments:
    training paths and test paths (list of strings): A list containing all file paths to read images from, stored as strings.
    zoom_level (int): The requested zoom level, with only 20, 21 and 22 being valid. Other zoom levels exist for our imagery, but the Apollo training set only contains those.
    
    Returns:
    Two lists (for train and test) of 896x896x3 images loaded from the file paths given.
    
    """
    
    if zoom_level not in [19, 20, 21, 22]:
        print('Invalid zoom level, please use 19, 20, 21 or 22 only')
        return None, None
    
    # Technically we can use the apollo zoom 22 image to get 256x256 results instead of 128x128, but we only need 128x128 as the cropping procedures on search tiles makes them 128x128 in the end
    file_zoom_level = 21

    training_imgs        = []
    test_imgs            = []
        
    for training_path in training_paths:
        im = plt.imread(training_path)
        if ((zoom_level in [20, 21, 22]) and im.shape[0] == 896):
            training_imgs += [im]
        if zoom_level == 19:
            if im.shape[0] == 1152 or im.shape[0] == 1408:
                im_crop = im[:1024, :1024, :]
                training_imgs += [im_crop]
            
    if len(training_imgs) != 0:   
        training_imgs = np.stack(training_imgs)
            
    for test_path in test_paths:
        im = plt.imread(test_path)
        if ((zoom_level in [20, 21, 22]) and im.shape[0] == 896):
            test_imgs += [im]
        if zoom_level == 19:
            if im.shape[0] == 1152 or im.shape[0] == 1408:
                im_crop = im[:1024, :1024, :]
                test_imgs += [im_crop]

    if len(test_imgs) != 0:
        test_imgs = np.stack(test_imgs)
        
    return training_imgs, test_imgs

def dilated_fcn_resnet(img_size, path_to_weights):
    """
    Small helper function to grab the model used for batch_inference()

    Arguments:
    img_size (tuple of integers): Size of the input to the model. Usually (896,896) or (1024,1024) when used by batch_inference()
    path_to_weights (string): Exact path of the .h5 weights for the keras model

    Returns:
    tfk.Model with the weights in path_to_weights loaded in
    """

    res50 = make_dilated_fcn_resnet_16s((img_size[0], img_size[1], 3), 34)
    res50.load_weights(path_to_weights)
    return tf.keras.Model(res50.input, res50.output)

def batch_inference(save_path, base_path = '/mnt/DATA/data/apollo_20190509/', path_to_weights = '/mnt/DATA/data_nagita/models/Q4_34class/model_13.h5', num_imgs = 30000, batches = 600, zoom_level = 21, blank_threshold = 0.05, save = True, verbose = True):
    
    """
    Run a set of Apollo images through a segmentation model, producing prediction rasters. Then, crop those prediction raster in to a given zoom level. The rasters are saved as a single group in save_path/all_zXX/ 
    
    Arguments:
    save_path (string): Base path where the new prediction rasters should be saved. 
    base_path (string): Location of the Apollo images.
    path_to_weights (string): Exact location of the Keras .h5 weights for the Apollo model
    num_imgs (int): Total number of images to load in. Note that as images are cropped for the required zoom level, more images will be produced. x49 for z22, x9 for z21, x1 for z20.
    batches (int): Split the process in to a number of batches to avoid GPU memory issues.
    zoom_level (int): Required zoom level, which the images are cropped to.
    blank_threshold (int): Only one blank image is allowed per batch to prevent an over-reliance on them. This threshold tunes how much data there needs to be for an image to be taken, from 0 to 1. Calculated by taking the mean pixel value across the whole image.
    save (boolean): Whether to save the results or not. Generally leave this as true unless debugging or just trying to gather results.
    
    Returns:
    None. The resulting images are saved to the specified location.
    
    """
    
    # The model which takes RGB images and produces prediction rasters - can be swapped out for other models which take the same image size.
    model = None
    if zoom_level == 19:
        model = dilated_fcn_resnet((1024,1024), path_to_weights)
    else:
        model = dilated_fcn_resnet((896,896), path_to_weights)
        
    batch_size = int(num_imgs/batches)
    training_num = 0
    test_num = 0
    
    if save_path[-1] == '/':
        save_path = save_path[:-1]
    
    os.system('mkdir -p {}'.format(save_path))
    
    os.system('mkdir {}/all_z{}/'.format(save_path, zoom_level))
    os.system('mkdir {}/all_z{}/train/'.format(save_path, zoom_level))
    os.system('mkdir {}/all_z{}/test/'.format(save_path, zoom_level))
    os.system('mkdir {}/all_z{}/train/class/'.format(save_path, zoom_level))
    os.system('mkdir {}/all_z{}/test/class/'.format(save_path, zoom_level))
    
    train_saved_all = 0
    train_total_all = 0
    test_saved_all = 0
    test_total_all = 0
    
    file_zoom_level = 21
    training_data_root = pathlib.Path(base_path+'{}/training/'.format(file_zoom_level))
    test_data_root = pathlib.Path(base_path+'{}/test/'.format(file_zoom_level))
        
    training_image_paths = []
    for i in training_data_root.glob('*'):
        image = list(i.glob('*.jpg'))[-1]
        training_image_paths.append(image)

    training_image_paths = sorted(training_image_paths)
    training_image_paths = [str(path) for path in training_image_paths]
        
    test_image_paths = []
    for i in test_data_root.glob('*'):
        image = list(i.glob('*.jpg'))[-1]
        test_image_paths.append(image)

    test_image_paths = sorted(test_image_paths)
    test_image_paths = [str(path) for path in test_image_paths]
    
    
    
    for b in tqdm_notebook(range(batches), disable = not verbose):
        
        offset = b*batch_size
        num_imgs = batch_size
        
        # Grab the batch images. These are full 7x7 images, not tiles
        training_imgs, test_imgs = dataset_from_apollo(zoom_level = zoom_level, 
                                                       training_paths = training_image_paths[offset:offset+num_imgs],
                                                       test_paths = test_image_paths[offset:offset+num_imgs])
        
        if len(training_imgs) == 0 and len(test_imgs) == 0:
            continue
        
        all_training_preds = []
        all_test_preds = []
        training_shape = 0
        test_shape = 0
        # Perform inference on batch
        if len(training_imgs) != 0:
            all_training_preds = model.predict(training_imgs/255, verbose=0)[:,:,:,:]
            training_shape = all_training_preds.shape[3]
        
        if len(test_imgs) != 0:
            all_test_preds = model.predict(test_imgs/255, verbose=0)[:,:,:,:]
            test_shape = all_test_preds.shape[3]
        
        train_saved = 0
        train_total = 0
        test_saved = 0
        test_total = 0
        
        
        for c in range(training_shape):
            
            if len(training_imgs) != 0:
                training_set = []
                training_preds = all_training_preds[:,:,:,c]
                training_cropped = crop_training_predictions(training_preds, zoom_level)


                # Only take non-blank tiles (with a single blank ones per batch)
                blank = True
                for i in range(len(training_cropped)):

                    if training_cropped[i][:,:].mean() < blank_threshold:
                        if blank == True:
                            blank = False
                            training_set.append((training_cropped[i][:,:]))
                    else:
                        training_set.append((training_cropped[i][:,:]))

                train_saved += len(training_set)
                train_total += len(training_cropped)
                
                if save:      
                    for i in range(len(training_set)):
                        im = PIL.Image.fromarray((training_set[i]*255).astype(np.uint8), mode = "L")
                        im.save('{}/all_z{}/train/class/{}.jpeg'.format(save_path, zoom_level, training_num))
                        training_num += 1
        
        for c in range(test_shape):
            if len(test_imgs) != 0:
                test_preds = all_test_preds[:,:,:,c]
                test_set  = []
                test_cropped = crop_training_predictions(test_preds, zoom_level)
                blank = True
                for i in range(len(test_cropped)):

                    # Only take non-blank tiles (with a single blank ones per batch)
                    if test_cropped[i][:,:].mean() < blank_threshold:
                        if blank == True:
                            blank = False
                            test_set.append((test_cropped[i][:,:]))
                    else:
                        test_set.append((test_cropped[i][:,:]))


                test_saved += len(test_set)
                test_total += len(test_cropped)
                if save: 
                    for i in range(len(test_set)):
                        im = PIL.Image.fromarray((test_set[i]*255).astype(np.uint8), mode = "L")
                        im.save('{}/all_z{}/test/class/{}.jpeg'.format(save_path, zoom_level, test_num))
                        test_num += 1
        
        train_saved_all += train_saved
        train_total_all += train_total
        test_saved_all += test_saved
        test_total_all += test_total
        if verbose:
            print('Saved {}/{} training and {}/{} testing'.format(train_saved, train_total, test_saved, test_total))
            print('In total saved {}/{} training and {}/{} testing'.format(train_saved_all, train_total_all, test_saved_all, test_total_all))

        
def crop_search_predictions(base_path, split = 5, categories = [1], batch_size = 10000, verbose = True):
    
    """
    Given a set of (prediction) tiles at a given zoom level, crop them to make a lot of overlapping tiles at the next zoom level. For example, a single 256x256 zoom 20 tile will be cropped to a bunch of 128x128 zoom 21 tiles with significant overlap. The amount of tiles is determined by the 'split' argument. On top of just cropping tiles one by one, adjacent tiles will be combined and cropped over their boundaries to provide a smooth transition.
    
    The cropping index works as follows for split = 5:
    
     _________________
    |  1  2  3  4  5  | 26
    |  6  7  8  9 10  | 27
    | 11 12 13 14 15  | 28
    | 16 17 18 19 20  | 29
    | 21 22 23 24 25  | 30
     _______________
      31 32 33 34 35    36
      
    This follows a similar structure for any value of split. For some tiles, one or more of (26,27,28,29,30), (31,32,33,34,35) or (36) may not exist as they are created by sharing with an adjacent tile (which won't exist at the edges of a dataset). 

    
    Arguments:
    base_path (string): Folder in which the original images are stored.
    split (int): How many overlapping images are produced per row/column. For example, split = 5 will produce 5*5=25 total overlapping images from a single tile. Additional cropping is performed inbetween tiles too.
    categories (list of ints): The prediction categories to use. 
    batch_size (int): Amount of images to process at a time
    verbose (boolean): Whether to print progress for the cropping of each category
    """
    
    if base_path[-1] == '/':
        base_path = base_path[:-1]
    
    for category in tqdm_notebook(categories, disable = not verbose, desc = "For each category"):
        
        length = len(glob.glob(base_path + '/' + str(int(category)) + '/*'))
        batch_verbose = True
        if length <= batch_size:
            batch_size = length
            batch_verbose = False
            if verbose:
                print('Set batch size to {} as length <= batch_size'.format(batch_size))
        
        for batch in tqdm_notebook(range(int(length/batch_size)), disable = not (verbose and batch_verbose), desc = "For each batch"):
        
            predictions, coordinates = load_predictions(base_path,
                                                        category = category,
                                                        cull = 1,
                                                        size = 256,
                                                        start = batch*batch_size,
                                                        total = batch_size,
                                                        verbose = False)

            coordinates_list = coordinates.tolist()

            # Width and height is equal
            size = predictions.shape[1]
            boundaries = []

            for i in range(2*split-1):
                boundaries.append(int((size*i)/(2*split-2)))

            # Construct the required folder structure so you don't have to do it in the loop
            path_name = base_path + '_cropped'
            os.system('mkdir -p {}/'.format(path_name))
            os.system('mkdir {}/{}/'.format(path_name, category))    

            for p in tqdm_notebook(range(predictions.shape[0]), disable = not verbose, desc = "Cropping predictions"):

                # Perform crops inside the given tile
                subplot_index = 1
                for row in range(split):
                    for col in range(split):
                        tile = predictions[p][boundaries[row]:boundaries[row+split-1],boundaries[col]:boundaries[col+split-1]]
                        im = PIL.Image.fromarray(np.array(tile).astype(np.uint8), mode = "L")
                        im.save('{}/{}/{}_{}_{}_{}.jpeg'.format(path_name, 
                                                             int(category), 
                                                             int(coordinates[p][0]), 
                                                             int(coordinates[p][1]),
                                                             int(split),
                                                             int(subplot_index)))
                        subplot_index += 1


                tx = coordinates[p][0]
                ty = coordinates[p][1]

                first_boundary = int(0.75 * size)
                second_boundary = int(0.25 * size)

                # if the right-hand tile also exists, do <split> overlapping tiles between the current one and the right-hand one
                if [tx+1,ty] in coordinates_list:
                    im_left = predictions[p]
                    im_right = predictions[coordinates_list.index([tx+1, ty])]

                    subplot_index = split*split + 1
                    for row in range(split):
                        tile_left = im_left[boundaries[row]:boundaries[row+split-1], (first_boundary):]
                        tile_right = im_right[boundaries[row]:boundaries[row+split-1], :(second_boundary)]
                        tile_concat = np.concatenate((tile_left, tile_right), axis = 1)

                        # Save the concatenated tile
                        im = PIL.Image.fromarray(np.array(tile_concat).astype(np.uint8), mode = "L")
                        im.save('{}/{}/{}_{}_{}_{}.jpeg'.format(path_name, 
                                                             int(category), 
                                                             int(coordinates[p][0]), 
                                                             int(coordinates[p][1]),
                                                             int(split),
                                                             int(subplot_index)))

                        subplot_index += 1

                # if (ty+1,tx) exists, do some overlapping ones between (ty,tx) and (ty+1,tx)
                if [tx, ty+1] in coordinates_list:
                    im_top = predictions[p]
                    im_bot = predictions[coordinates_list.index([tx, ty+1])]

                    subplot_index = split*(split+1) + 1
                    for col in range(split):
                        tile_top = im_top[(first_boundary):, boundaries[col]:boundaries[col+split-1]]
                        tile_bot = im_bot[:(second_boundary), boundaries[col]:boundaries[col+split-1]]
                        tile_concat = np.concatenate((tile_top, tile_bot), axis = 0)

                        # Save the concatenated tile
                        im = PIL.Image.fromarray(np.array(tile_concat).astype(np.uint8), mode = "L")
                        im.save('{}/{}/{}_{}_{}_{}.jpeg'.format(path_name, 
                                                             int(category), 
                                                             int(coordinates[p][0]), 
                                                             int(coordinates[p][1]), 
                                                             int(split),
                                                             int(subplot_index)))

                        subplot_index += 1

                # if (ty+1,tx+1) exists, do an overlapping one between all 4
                if [tx+1, ty+1] in coordinates_list:

                    im_tl = predictions[p] 
                    im_tr = predictions[coordinates_list.index([tx+1, ty])]
                    im_bl = predictions[coordinates_list.index([tx, ty+1])]
                    im_br = predictions[coordinates_list.index([tx+1, ty+1])]

                    tile_tl = im_tl[first_boundary:, first_boundary:]
                    tile_tr = im_tr[first_boundary:, :second_boundary]
                    tile_bl = im_bl[:second_boundary, first_boundary:]
                    tile_br = im_br[:second_boundary, :second_boundary]
                    top = np.concatenate((tile_tl, tile_tr), axis = 1)
                    bottom = np.concatenate((tile_bl, tile_br), axis = 1)
                    tile_concat = np.concatenate((top, bottom), axis = 0)
                    # Save the concatenated tile
                    im = PIL.Image.fromarray(np.array(tile_concat).astype(np.uint8), mode = "L")
                    im.save('{}/{}/{}_{}_{}_{}.jpeg'.format(path_name,
                                                         int(category),
                                                         int(coordinates[p][0]),
                                                         int(coordinates[p][1]),
                                                         int(split),
                                                         int(split*(split+2) + 1)))
        
                
def copy_n_files(base_path = '/mnt/DATA/data_filip/apollo_predictions/', n_train = 50000, n_test = 50000, zoom_level = 20, category = 'all', verbose = True):
    
    """
    Take a full set of images and extract a random N images in to a new folder. This is useful for training, where the full dataset contains millions of images but we wish to create smaller datasets.
    
    Arguments:
    base_path (string): The directory where the original images reside, not including the folder name.
    n_train (int): How many random training images to extract.
    n_test (int): How many random test images to extract.
    zoom_level (int): A portion of the folder name.
    category (string): The other portion of the folder name.
    verbose (boolean): Whether to show tqdm progress bars.
    
    Returns:
    None. All images are stored in a new folder and nothing is returned.
    """
    
    names_train = []
    names_test = []
    
    # Take the full list of (sorted) file names and shuffle it
    for i in tqdm_notebook(sorted(glob.glob('{}{}_z{}/train/class/*'.format(base_path, category, zoom_level))), disable = not verbose, desc = "Gathering training locations"):
        names_train.append(i)
    for i in tqdm_notebook(sorted(glob.glob('{}{}_z{}/test/class/*'.format(base_path, category, zoom_level))), disable = not verbose, desc = "Gathering testing locations"):
        names_test.append(i)

    random.shuffle(names_train)
    random.shuffle(names_test)

    # Create necessary file structure
    os.system('rm -r {}{}_{}_z{}/'.format(base_path, category, n_train, zoom_level))
    os.system('mkdir -p {}{}_{}_z{}/'.format(base_path, category, n_train, zoom_level))
    os.system('mkdir {}{}_{}_z{}/train/'.format(base_path, category, n_train, zoom_level))
    os.system('mkdir {}{}_{}_z{}/test/'.format(base_path, category, n_train, zoom_level))
    os.system('mkdir {}{}_{}_z{}/train/class/'.format(base_path, category, n_train, zoom_level))
    os.system('mkdir {}{}_{}_z{}/test/class/'.format(base_path, category, n_train, zoom_level))
    
    # From the shuffled list, take the first N elements and copy them over to the new folder 
    for i in tqdm_notebook(range(n_train), disable = not verbose, desc = "Copying training images"):
        os.system('cp {} {}{}_{}_z{}/train/class/'.format(names_train[i], base_path, category, n_train, zoom_level))
        
    for i in tqdm_notebook(range(n_test), disable = not verbose, desc = "Copying testing images"):
        os.system('cp {} {}{}_{}_z{}/test/class/'.format(names_test[i], base_path, category, n_train, zoom_level))

        
def crop_tx_ty(tx, ty, split, crop_num, z, datestr):
    
    """
    Used when displaying search results, this function will return an image (not a prediction) of the correct cropped position based on coordinates and a crop index. As the search data actually contains overlapping cropped tiles acquired from the previous zoom level (z-1), the coordinates are stored in that zoom level. For example, a search over z21 will actually use z20 coordinates with a cropping index. 
    
    The cropping index works as follows for split = 5:
    
     _________________
    |  1  2  3  4  5  | 26
    |  6  7  8  9 10  | 27
    | 11 12 13 14 15  | 28
    | 16 17 18 19 20  | 29
    | 21 22 23 24 25  | 30
     _______________
      31 32 33 34 35    36
      
    This follows a similar structure for any value of split. For some tiles, one or more of (26,27,28,29,30), (31,32,33,34,35) or (36) may not exist as they are created by sharing with an adjacent tile (which won't exist at the edges of a dataset). 
    
    Arguments:
    tx (int): x coordinate of the tile in tile coordinate format.
    ty (int): y coordinate of the tile in tile coordinate format.
    split (int): Amount of overlapped cropping that was performed on the tile. For example, split = 5 will result in 25 overlapping tiles from a single z-1 tile.
    crop_num (int): Crop index of the tile based on the amount of overlap cropping (split).
    z (int): Zoom level that the search was conducted at. In the previous example, this would be z21.
    datestr (str): Date at which to grab the tile. 
    
    Returns:
    im_cropped (NxNx3 numpy array, 0-255): Resultant image after cropping to the correct location. As the API returns 256x256x3 images, this should be 128x128x3.
    """
    
    crop_num = int(crop_num)
    tile_req = TileRequester(cache_dir = 'cache/tile_requester/') 
    im = np.array(tile_req.get_tile_tc(z-1, tx, ty, datestr))

    size = im.shape[0]
    first_boundary = int(0.75 * size)
    second_boundary = int(0.25 * size)
    boundaries = []
    for i in range(2*split-1):
        boundaries.append(int((size*i)/(2*split-2)))

    
    if crop_num <= split*split:
        row = int(np.ceil(crop_num/split)) - 1
        col = crop_num%split - 1
        if col == -1:
            col = split - 1

        im_cropped = im[boundaries[row]:boundaries[row+split-1],
                        boundaries[col]:boundaries[col+split-1], 
                        :]
    
    # Right-hand part
    elif crop_num <= split*(split+1):
        row = int((crop_num-1)%split)
        right_im = np.array(tile_req.get_tile_tc(z-1, tx+1, ty, datestr))
        tile_left = im[boundaries[row]:boundaries[row+split-1], (first_boundary):]
        tile_right = right_im[boundaries[row]:boundaries[row+split-1], :(second_boundary)]
        
        im_cropped = np.concatenate((tile_left, tile_right), axis = 1)
        
    # Bottom part
    elif crop_num <= split*(split+2):
        col = int((crop_num-1)%split)
        bottom_im = np.array(tile_req.get_tile_tc(z-1, tx, ty+1, datestr))
        tile_top = im[(first_boundary):, boundaries[col]:boundaries[col+split-1]]
        tile_bot = bottom_im[:(second_boundary), boundaries[col]:boundaries[col+split-1]]
        im_cropped = np.concatenate((tile_top, tile_bot), axis = 0)
    
    # Bottom-right part
    elif crop_num == split*(split+2) + 1:
        right_im = np.array(tile_req.get_tile_tc(z-1, tx+1, ty, datestr))
        bottom_im = np.array(tile_req.get_tile_tc(z-1, tx, ty+1, datestr))
        bottom_right_im = np.array(tile_req.get_tile_tc(z-1, tx+1, ty+1, datestr))
        
        tile_tl = im[first_boundary:, first_boundary:]
        tile_tr = right_im[first_boundary:, :second_boundary]
        tile_bl = bottom_im[:second_boundary, first_boundary:]
        tile_br = bottom_right_im[:second_boundary, :second_boundary]
        
        top = np.concatenate((tile_tl, tile_tr), axis = 1)
        bottom = np.concatenate((tile_bl, tile_br), axis = 1)

        im_cropped = np.concatenate((top, bottom), axis = 0)
        
    return im_cropped

def crop_tx_ty_prediction(tx, ty, split, crop_num, z, datestr, category = 1):
    
    """
    Same as crop_tx_ty but for prediction rasters instead of RGB images.
    
    Used when displaying search results, this function will return a prediction raster of the correct cropped position based on coordinates and a crop index. As the search data actually contains overlapping cropped tiles acquired from the previous zoom level (z-1), the coordinates are stored in that zoom level. For example, a search over z21 will actually use z20 coordinates with a cropping index. 
    
    The cropping index works as follows for split = 5:
    
     _________________
    |  1  2  3  4  5  | 26
    |  6  7  8  9 10  | 27
    | 11 12 13 14 15  | 28
    | 16 17 18 19 20  | 29
    | 21 22 23 24 25  | 30
     _______________
      31 32 33 34 35    36
      
    This follows a similar structure for any value of split. For some tiles, one or more of (26,27,28,29,30), (31,32,33,34,35) or (36) may not exist as they are created by sharing with an adjacent tile (which won't exist at the edges of a dataset). 
    
    Arguments:
    tx (int): x coordinate of the tile in tile coordinate format.
    ty (int): y coordinate of the tile in tile coordinate format.
    split (int): Amount of overlapped cropping that was performed on the tile. For example, split = 5 will result in 25 overlapping tiles from a single z-1 tile.
    crop_num (int): Crop index of the tile based on the amount of overlap cropping (split).
    z (int): Zoom level that the search was conducted at. In the previous example, this would be z21.
    datestr (str): Date at which to grab the tile. 
    category (int): Prediction category to grab for the raster.
    
    Returns:
    im_cropped (NxN numpy array, 0-255): resultant image after cropping to the correct location. As the API returns 256x256 images, this should be 128x128.
    """
    
    crop_num = int(crop_num)
    pred_req = PredictionRequester() 
    
    url = ("https://api.nearmap.com/coverage/v2/coord/{}/{}/{}?apikey={}&until={}&limit=1".format(z-1, tx, ty, os.environ["API_KEY"], datestr))
    response = requests.get(url).json()
    survey = response["surveys"][0]["id"]

    im = np.array(pred_req.get_preds_tc_with_survey(z-1, tx, ty, survey, classes = [category]))

    size = im.shape[0]
    first_boundary = int(0.75 * size)
    second_boundary = int(0.25 * size)
    boundaries = []
    for i in range(2*split-1):
        boundaries.append(int((size*i)/(2*split-2)))

    if crop_num <= split*split:
        row = int(np.ceil(crop_num/split)) - 1
        col = crop_num%split - 1
        if col == -1:
            col = split - 1

        im_cropped = im[boundaries[row]:boundaries[row+split-1],
                        boundaries[col]:boundaries[col+split-1], 
                        :]
    
    # Right-hand part
    elif crop_num <= split*(split+1):
        row = int((crop_num-1)%split)
        right_im = np.array(pred_req.get_preds_tc_with_survey(z-1, tx+1, ty, survey, classes = [category]))
        tile_left = im[boundaries[row]:boundaries[row+split-1], (first_boundary):]
        tile_right = right_im[boundaries[row]:boundaries[row+split-1], :(second_boundary)]
        
        im_cropped = np.concatenate((tile_left, tile_right), axis = 1)
        
    # Bottom part
    elif crop_num <= split*(split+2):
        col = int((crop_num-1)%split)
        bottom_im = np.array(pred_req.get_preds_tc_with_survey(z-1, tx, ty+1, survey, classes = [category]))
        tile_top = im[(first_boundary):, boundaries[col]:boundaries[col+split-1]]
        tile_bot = bottom_im[:(second_boundary), boundaries[col]:boundaries[col+split-1]]
        im_cropped = np.concatenate((tile_top, tile_bot), axis = 0)
    
    # Bottom-right part
    elif crop_num == split*(split+2) + 1:
        right_im = np.array(pred_req.get_preds_tc_with_survey(z-1, tx+1, ty, survey, classes = [category]))
        bottom_im = np.array(pred_req.get_preds_tc_with_survey(z-1, tx, ty+1, survey, classes = [category]))
        bottom_right_im = np.array(pred_req.get_preds_tc_with_survey(z-1, tx+1, ty+1, survey, classes = [category]))
        
        tile_tl = im[first_boundary:, first_boundary:]
        tile_tr = right_im[first_boundary:, :second_boundary]
        tile_bl = bottom_im[:second_boundary, first_boundary:]
        tile_br = bottom_right_im[:second_boundary, :second_boundary]
        
        top = np.concatenate((tile_tl, tile_tr), axis = 1)
        bottom = np.concatenate((tile_bl, tile_br), axis = 1)

        im_cropped = np.concatenate((top, bottom), axis = 0)
        
    return im_cropped
    
    
def raw_moment(data, i_order, j_order):
    
    """
    This function and moments_cov() were taken from https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    They are used in the rotate_tile() function
    
    Arguments:
    data (numpy array): Image for which the raw moments are being calculated. If used as part of rotate_tile(), this will be a 128x128x1 or 256x256x1 prediction raster.
    i_order (int): Power of x in the raw moments equation.
    j_order (int): Power of y in the raw moments equation.  
    
    Returns:
    Value of the raw image moment (float).
    """
    
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]
    return (data * x_indicies**i_order * y_indices**j_order).sum()

def moments_cov(data):
    
    """
    This function and raw_moment() were taken from https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    They are used in the rotate_tile() function
    
    Arguments:
    data (numpy array): Image for which the raw moments are being calculated. If used as part of rotate_tile(), this will be a 128x128x1 or 256x256x1 prediction raster.
    
    Returns:
    Moments covariance array (numpy array) which is used to calculate the required rotation in rotate_tile()
    """
    
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum if data_sum != 0.0 else 0.0
    y_centroid = m01 / data_sum if data_sum != 0.0 else 0.0
        
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum if data_sum != 0.0 else 0.0
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum if data_sum != 0.0 else 0.0
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum if data_sum != 0.0 else 0.0
    cov = np.array([[u20, u11], [u11, u02]])
    return cov
    
    
def rotate_tile(im, tolerance = 1, contour_level = 100, cutoff_tolerance = 0.6, padding_mode = 'constant', rotation_mode = 'nearest', verbose = False):
    
    """
    Given a prediction raster tile, rotate it such that it is aligned to the nearest right angle using either hough transforms or raw image moments. Hough transforms are attempted first as they work better on square shapes like roofs, but if it fails then raw image moments are used, which work better on blobs such as pools. Images are rotated about their centroid, not just the centre of the image.
    
    Arguments:
    im (numpy array, 0-255): NxN or 1xNxN numpy array representing the image to be rotated. 
    tolerance (int): Tolerance for the skimage function approximate_polygon(). "Maximum distance from original points of polygon to approximated polygonal chain. If tolerance is 0, the original coordinate array is returned."
    contour_level (int): Contour level for the skimage find_contours() function. "Value along which to find contours in the array"
    cutoff_tolerance (int): Tolerance from 0 to 1 of how much of the original image needs to remain after rotation for it to be allowed. A pool in the corner of the image may be completely lost after rotation, so we want to avoid that. 0.6 = 60% of the original image needs to remain to be allowed.
    padding_mode (string): Mode to use for np.pad() when performing a rotation around the centroid.
    rotation_mode (string): Mode to use for scipy.ndimage.rotate() when performing a rotation around the centroid.
    verbose (boolean): Whether or not to print additional information for debugging.
    
    Returns:
    im_rotated_cropped (numpy array, 0-255): NxN image, representing the original image but rotated about its centre of mass to be aligned to the nearest right angle.
    
    """
        
    # Images can be (1,N,N) depending on where they came from
    if len(im.shape) == 3:
        im = im[0]   
        
    polygons = []
    
    was_01 = False
    # If the image is actually 0-1 (or thereabouts) and has enough features, make it 0-255
    if im.max() < 2 and im.mean() > 0.1:
        was_01 = True
        im = im.copy() * 255
    
    if im.mean() < 25:
        return im
    
    # First try to rotate the whole thing using a hough transform. If this fails, try the more blob-based raw image moments approach.
    hough = get_hough_transform_angle(im, 0.1)
    if hough is not None:
        if verbose == True:
            print('Used hough with {} degrees'.format(-np.degrees(hough)))
        avg = -np.degrees(hough)
    
    else:
        # Split the image into n polygons
        for contour in find_contours(im, contour_level):
            coords = approximate_polygon(contour, tolerance = tolerance)
            polygons.append(coords)

        if len(polygons) > 5 or len(polygons) < 1: 
            if was_01 == True:
                return im/255
            else:
                return im

        # Turn those polygons in to individual masks
        masks = []
        for p in range(len(polygons)):
            img = np.zeros((128, 128))
            rr, cc = polygon(polygons[p][:,0], polygons[p][:,1], (128, 128))
            img[rr,cc] = 255
            masks.append(img)

        # Figure out how to rotate the whole image based on the rotation needed for each polygon
        rotations = []
        for m in masks:

            cov = moments_cov(m)
            if (True in np.isnan(cov)) or (True in np.isinf(cov)):
                rotations.append(0)
                continue
            evals, evecs = np.linalg.eig(cov)
            sort_indices = np.argsort(evals)[::-1]
            x_v1, y_v1 = evecs[:, sort_indices[0]]
            x_v2, y_v2 = evecs[:, sort_indices[1]]

            m = y_v1/x_v1 if x_v1 != 0.0 else 0.0
            theta = np.arctan(m) 
            rotations.append(np.degrees(theta))
        
        # If the polygons need very different rotations, just discard it
        if len(masks) > 2 and np.std(rotations) > 50:
            if was_01 == True:
                return im/255
            else:
                return im
        
        # If the polygons need reasonably different rotations, take the rotation necessary to align the largest polygon
        if max(rotations) - min(rotations) > 20:
            means = []
            for m in masks:
                means.append(m.mean())
            avg = rotations[means.index(max(means))]
            
        else:
            avg = np.mean(rotations)
    
    
    # Rotate the image about its centroid by padding the image so the centroid is the middle, rotating and then cropping back at an offset
    centroid = scipy.ndimage.measurements.center_of_mass(im)
    
    if verbose == True:
        print('Centroid:', centroid)
    
    padY = [im.shape[1] - int(centroid[0]), int(centroid[0])]
    padX = [im.shape[0] - int(centroid[1]), int(centroid[1])]
    im_padded = np.pad(im, [padY, padX], padding_mode)
    im_rotated = scipy.ndimage.rotate(im_padded, avg, reshape=False, mode = rotation_mode)
    
    im_rotated_cropped = im_rotated[padY[0] : -padY[1], padX[0] : -padX[1]]
    
    # Don't cut off too much of the original image.
    # This will happen when the feature is in the corner or edge of the image
    if im_rotated_cropped.mean()/im.mean() < cutoff_tolerance:
        if was_01 == True:
            return im/255
        else:
            return im
    
    if was_01 == True:
        return im_rotated_cropped/255
    else:
        return im_rotated_cropped    


def rotate_model_tiles(base_path, tolerance = 10, contour_level = 100, cutoff_tolerance = 0.5, verbose = True):
    
    """
    Rotate a set of images based on a file path, using the rotate_tile() function. All arguments except base_path are just passed on to the rotate_tile() function.
    This is designed for use with model training/testing folders, as it has /train and /test etc. subfolders built in.
    
    Arguments:
    base_path (string): Rxact directory where the images are located.
    tolerance (int): Tolerance for the skimage function approximate_polygon(). "Maximum distance from original points of polygon to approximated polygonal chain. If tolerance is 0, the original coordinate array is returned."
    contour_level (int): Contour level for the skimage find_contours() function. "Value along which to find contours in the array"
    cutoff_tolerance (int): Tolerance from 0 to 1 of how much of the original image needs to remain after rotation for it to be allowed. A pool in the corner of the image may be completely lost after rotation, so we want to avoid that. 0.6 = 60% of the original image needs to remain to be allowed.
    
    Returns:
    None. The images are saved in another directory, which is the base path with '_aligned' appended to it. 
    """
    
    if base_path[-1] == '/':
        base_path = base_path[:-1]
    os.system('mkdir -p {}_aligned'.format(base_path))
    os.system('mkdir {}_aligned/train'.format(base_path))
    os.system('mkdir {}_aligned/test'.format(base_path))
    os.system('mkdir {}_aligned/train/class'.format(base_path))
    os.system('mkdir {}_aligned/test/class'.format(base_path))
    
    for i in tqdm_notebook(glob.glob(base_path + '/train/class/*.jpeg'), disable = not verbose, desc = "Rotating training tiles"):
        
        image_name = i.split('/')[-1]
        
        im = plt.imread(i)
        im_rotated = rotate_tile(im, tolerance, contour_level, cutoff_tolerance, verbose = False)
        
        im_rotated_pil = PIL.Image.fromarray(im_rotated, mode = "L")
        im_rotated_pil.save(base_path + '_aligned/train/class/' + image_name)
    
    for i in tqdm_notebook(glob.glob(base_path + '/test/class/*.jpeg'), disable = not verbose, desc = "Rotating testing tiles"):
        
        image_name = i.split('/')[-1]
        
        im = plt.imread(i)
        im_rotated = rotate_tile(im, tolerance, contour_level, cutoff_tolerance, verbose = False)
        
        im_rotated_pil = PIL.Image.fromarray(im_rotated, mode = "L")
        im_rotated_pil.save(base_path + '_aligned/test/class/' + image_name)


def rotate_search_tiles(base_path, categories, tolerance = 10, contour_level = 100, cutoff_tolerance = 0.5, start = 0, verbose = True):    
    
    """
    Rotate a set of images based on a file path, using the rotate_tile() function. All arguments except base_path are just passed on to the rotate_tile() function.
    If you wish to batch rotate tiles for model training (which are in subfolders /train, /test, etc, use rotate_model_tiles())
    
    Arguments:
    base_path (string): Directory where the images are located, excluding the category folder.
    categories (int): The category folders the images are in.
    tolerance (int): Tolerance for the skimage function approximate_polygon(). "Maximum distance from original points of polygon to approximated polygonal chain. If tolerance is 0, the original coordinate array is returned."
    contour_level (int): Contour level for the skimage find_contours() function. "Value along which to find contours in the array."
    cutoff_tolerance (int): Tolerance from 0 to 1 of how much of the original image needs to remain after rotation for it to be allowed. A pool in the corner of the image may be completely lost after rotation, so we want to avoid that. 0.6 = 60% of the original image needs to remain to be allowed.
    start (int): Optional starting index, so you can skip some elements if you wish
    verbose (boolean): Whether to print tqdm progress bars.
    
    Returns:
    None. The images are saved in another directory, which is the base path with '_aligned' appended to it, in their respective category folder. 
    """
    
    for category in tqdm_notebook(categories, disable = not verbose, desc = "For each category"):
        if base_path[-1] == '/':
            base_path = base_path[:-1]

        os.system('mkdir -p {}_aligned'.format(base_path))
        os.system('mkdir {}_aligned/{}'.format(base_path, int(category)))

        for i in tqdm_notebook(glob.glob(base_path + '/' + str(int(category)) + '/*.jpeg')[start:], disable = not verbose, desc = "Rotating predictions"):
            
            image_name = i.split('/')[-1]
            
            # If the previous run was stopped halfway through or something similar, this will skip through the stuff that's already been made
            if os.path.isfile(base_path + '_aligned/' + str(int(category)) + '/' + image_name):
                continue

            im = plt.imread(i)
            im_rotated = rotate_tile(im, tolerance, contour_level, cutoff_tolerance, verbose = False)


            im_rotated_pil = PIL.Image.fromarray(im_rotated, mode = "L")
            im_rotated_pil.save(base_path + '_aligned/' + str(int(category)) + '/' + image_name)
        
        
def tfidf(base_path, tf_threshold):
    
    """
    Calculate the inter-document frequency for a given set of prediction raster tiles, for all categories in a given directory
    
    Arguments:
    base_path (string): Directory where the images are located, excluding the category folder.
    tf_threshold (int): Threshold value (0-255) to use when determining if a single pixel should be used for the average TF calculation.
    
    Returns:
    tfidfs (dictionary): Dict mapping category (integer) to a list of [average TF, IDF] for that category.
    """
    
    if base_path[-1] == '/':
        base_path = base_path[:-1]
    
    category_list = glob.glob(base_path + '/*')
    tfidfs = {}
    
    for c in tqdm_notebook(category_list):
        
        category = c.split('/')[-1]
        
        file_list = glob.glob(base_path + '/' + str(int(category)) + '/*')
        
        # Total number of images
        num_images = len(file_list)

        avg_tf = 0
        num_category = 0

        # Number of images with the given category in it
        # Also calculate average TF for each image while we're at it
        for i in file_list:
            im = plt.imread(i)

            if im.mean() > 2:
                num_category += 1

            avg_tf += len(np.argwhere(im > tf_threshold))/(im.shape[0] * im.shape[1])

        avg_tf /= len(file_list)
        if num_category != 0:  
            idf = math.log(num_images/num_category)
        else:
            idf = 0
        
        tfidfs[category] = [avg_tf, idf]
        
    return tfidfs


def rotate_about_centroid(im, angle):
    """
    Rotate a given image about its centroid rather than just about the middle. Useful for things like swimming pools and solar panels
    
    Arguments:
    im (2D numpy array): Image to be rotated, in 2D format. 
    angle (float): Angle of rotation for the image.
    
    Returns:
    rotated_cropped (2D numpy array): The input image but rotated about its centroid.
    """
    
    if im.mean() < 0.01:
        centroid = (float('nan'),float('nan'))
    else:
        centroid = scipy.ndimage.measurements.center_of_mass(im)

    if np.isnan(centroid[0]) or np.isnan(centroid[1]):
        rotated_cropped = im
    else:
        padY = [im.shape[1] - int(centroid[0]), int(centroid[0])]
        padX = [im.shape[0] - int(centroid[1]), int(centroid[1])]
        im_padded = np.pad(im, [padY, padX], 'edge')
        rotated = scipy.ndimage.rotate(im_padded, angle, reshape = False)
        rotated_cropped = rotated[padY[0] : -padY[1], padX[0] : -padX[1]]
    
    return rotated_cropped


def encode_predictions(base_path, predictions_path, coordinates_path, encoder, categories, verbose = True, start = 0):
    """
    Run all the given prediction tiles through the encoder, saving the resulting encodings as individual numpy arrays per category
    
    Arguments:
    base_path (string): Directory in which prediction tiles are stored, not including the category folder.
    predictions_path (string): Location (excluding file name) to save each encoded predictions numpy array, per category.
    coordinates_path (string): Location (including file name) to save the resultant coordinates numpy array.
    encoder (keras model): Model that takes in the images and produces a resultant encoding. This can be any keras model that takes in a greyscale image as input and produces a 1D vector.
    categories (list of ints): Folders in which the images are stored, appended to the base_path.
    verbose (boolean): Whether or not to print progress.
    start (int): Optional starting index, so you can skip some elements if you wish.
    
    Returns:
    None
    """
    
    os.system('mkdir -p {}'.format(predictions_path))
    os.system('mkdir -p {}'.format(coordinates_path[:-len(coordinates_path.split('/')[-1])]))
    
    for category in tqdm_notebook(categories, disable = not verbose, desc = "For each category"):
        if base_path[-1] == '/':
            base_path = base_path[:-1]
            
        if predictions_path[-1] == '/':
            predictions_path = predictions_path[:-1]

        file_list = glob.glob(base_path + "/" + str(category) + "/*.jpeg")[start:]
        batches = int(len(file_list[start:])/25)
        encoded_pred = []

        for f in tqdm_notebook(file_list, disable = not verbose, desc = "Encode predictions"):

            predictions = load_predictions_from_file_list([f], 
                                                       get_coordinates = False,
                                                       verbose = False)

            encoded_pred.append(encoder(np.expand_dims((predictions/255), axis=3))[0].numpy())
        
        encoded_pred = np.array(encoded_pred)[:,0,:]
    
        # Coordinates are the same for all categories so only do it once
        if category == categories[0]:
            coordinates = load_predictions_from_file_list(file_list, get_predictions = False, verbose = False)
            np.save(coordinates_path, coordinates)
        
        np.save(predictions_path + '/' + 'c{}.npy'.format(category), encoded_pred)
    
    
def create_index(encoding_path, save_path, categories, ef_construction = 50000, M = 100, num_threads = 8, index_space = 'ip'):
    
    """
    Takes in a (location for a) numpy array of encoded predictions and creates a HNSW search index that can be used to efficiently search over them.
    More detailed information about the hyperparameters is available here: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    
    Arguments:
    encoding_path (string): Exact path of the encoded layers as a numpy array.
    save_path (string): Location where the index should be saved.
    ef_construction (int): The size of the dynamic list for the nearest neighbors (used during the search). A higher value will increase build time and index accuracy.
    M (int): The number of bi-directional links created for every new element during construction. The size of the index is roughly M * 8-10 bytes per stored element.
    num_threads (int): How many CPU threads to use for index creation. Multiple create_index() calls are not thread safe as multiple hnswlib.Index.add_items() calls are not thread safe.
    
    Returns:
    None.
    """
    
    print('If you are using large parameter values, this will take many hours. ef_construction = 50,000 and M=100 takes 12 hours to make 1.5M 96-dim elements')
    if encoding_path[-1] == '/':
        encoding_path = encoding_path[:-1]
    if save_path[-1] == '/':
        save_path = save_path[:-1]
    
    os.system('mkdir -p {}'.format(save_path))
    
    for c in categories:
        f = encoding_path + '/c' + str(int(c)) + '.npy'
        encoded_pred = np.load(f)
        file_name = f.split('/')[-1][:-4] # Just the file name, not including .npy
        index = hnswlib.Index(space = index_space, dim = encoded_pred.shape[1])
        index.init_index(encoded_pred.shape[0], ef_construction, M)
        index.add_items(encoded_pred, num_threads = 8)

        index.save_index(save_path + '/{}.idx'.format(file_name))

        
def roll_blackout(arr, roll_amount, axis = 0):
    
    """
    Extends the np.roll() functionality to also write zeros where the image was rolled away from, rather than looping around elements.
    
    Arguments:
    arr (numpy array): Input image with dimensions (categories, size, size). All categories are rolled at the same time.
    roll_amount (int): How many pixels to roll (positive or negative).
    axis (int): Axis in which to roll.
    
    Returns:
    rolled (numpy array): Same dimensions and values as the input array, just rolled with the given specifications.
    """
    
    if axis > 1 or roll_amount == 0:
        return arr
    
    rolled = np.roll(arr, roll_amount, axis = (axis+1))
    
    if axis == 1:
        if roll_amount > 0:
            rolled[:, :, :roll_amount] = 0
        else:
            rolled[:, :, roll_amount:] = 0
    elif axis == 0:
        if roll_amount > 0:
            rolled[:, :roll_amount, :] = 0
        else:
            rolled[:, roll_amount:, :] = 0
    
    return rolled


def jiggle_image(image, encoder, encoding_size = 96, layers = 2):
        
    """
    Given an image, translate it left/right/up/down and produce encodings for all of those images.
    
    Arguments:
    image (numpy array): float32 array representing an image of dimensions (categories, 128, 128).
    encoder (keras model): Keras model, as part of the autoencoder, which will produce an encoding of a given input image.
    encoding_size (int): Amount of numbers in the encoding vector.
    layers (int): How much translation to apply. layers = 1 applies +-16px in both axes, 2 applies +-16px and +-8px, 3 applies +-16px, +-8px and +-24px and 4 applies +-16px, +-8px, +-24px and +-4px. 
    
    Returns:
    jiggled_encodings (numpy array): Array of encoding vectors for each of the jiggled images. This is of dimension (n, encoding_size), where n is the amount of jiggled images produced.
    """
    
    if layers > 4:
        layers = 4
        
    # Sorted specifically this way due to the decreasing impact on search results
    jiggles = [16, 8, 24, 4]    
    
    num_categories = image.shape[0]
    
    rolled_images = []
    rolled_images.append(image[:,:,:])
        
    j = 1
    
    # Roll each image, translating it in various directions. This will make a (n, categories, 128, 128) array, where n is the amount of rolling done
    for roll_amount in (jiggles[:layers] + [x * -1 for x in jiggles][:layers]):
        for a in [0, 1]:
            rolled = roll_blackout(image[:,:,:], roll_amount, axis = a)
            rolled_images.append(rolled)
                
            j += 1
    rolled_images = np.array(rolled_images)
    
    # Flatten the array so that its just a bunch of 128 x 128 images to be passed through the model efficiently
    rolled_images = np.reshape(rolled_images, 
                               (rolled_images.shape[0] * rolled_images.shape[1], 
                                rolled_images.shape[2], 
                                rolled_images.shape[3]))
    
    rolled_images = np.expand_dims(rolled_images, axis = 3)
    jiggled_encodings = encoder.predict(rolled_images, batch_size = 20)[0]
    
    # Split the resultant encodings so its shape is (n, categories, encoding_size) instead of ((n x categories), encoding_size)
    jiggled_encodings = np.array(np.split(jiggled_encodings, jiggled_encodings.shape[0]/num_categories))
    return jiggled_encodings


def search(preds_aligned, encoder, index_directory, coordinates_directory, dataset_datestr, zoom_level, classes_cut, weights_cut, search_ef, num_nearest, rotations, index_space, index_dim, show_images):
    
    
    """
    This function is used by search_location() and search_image() and shouldn't be called manually. 
    
    Arguments: 
    preds_aligned (numpy array): 128x128xChannels image of prediction rasters
    encoder (Keras model): Encoder from the variational autoencoder model, used to produce an encoding vector from a given image.
    index_directory (string): Folder in which the indexes are stored. One index is made per category, and they are stored in a single folder. Each index has the name "c<category>.idx".
    coordinates_directory (string): Exact location of the coordinates numpy array.
    dataset_datestr (string): Date for which to grab the resultant similar images, in YYYY-MM-DD format.
    zoom_level (int): Zoom level over which to perform the search.
    classes_cut (list of ints): Prediction categories to use for the search, filtered such that classes corresponding to 0 weight are removed.
    weights_cut (list of floats): Weights for each category, following the same order as the previous 'categories' argument. 0 weights are removed.
    search_ef (int): Search parameter. Higher values lead to a higher quality search but longer search times. Optimal values are around 1000-5000, any higher will cause quite long searches.
    num_nearest (int): How many similar locations to return.
    rotations (int): How many rotations to perform when searching. It's a good idea to just leave this at 4 (ie 0, 90, 180, 270 degrees).
    index_space (string): Search space to use for the hnswlib search. Only 'l2', 'ip' (inner product) and 'cosine' are supported. If other spaces are necessary, consider migrating to the hnsw method in nmslib instead.
    index_dim (int): Dimensionality of the encoded predictions in the search index.
    show_images (boolean): Whether or not to show the query and result images using matplotlib.pyplot
    
    Returns:
    (result_coordinates, result_images, result_ids) tuple
    
    result_coordinates (list of tuples): A list which holds the tile coordinates (tx, ty, split_amount, crop_num) as tuples. These are the nearest neighbours, and the amount returned depends on num_nearest. The values are the same as what's used by crop_tx_ty(), but are handy if the locations need to be used elsewhere
    result_images (list of np arrays): A list of 128x128x3 numpy arrays representing the similar RGB images. The amount returned depends on num_nearest.
    result_ids (list of ints): A list which contains the IDs of each of the top N results, as they are stored in the relevant index and encoded predictions list. 
    """
    
    coordinates = np.load(coordinates_directory)
    index = hnswlib.Index(space = index_space, dim = index_dim)
    
    # Jiggle the tile in order to account for some slight translation differences. 
    # Also rotate the tile 0/90/180/270 degrees
    # This means that we search many different tiles and get the best matches
    im_query_augmented = None
    for r in range(0,360,int(360/rotations)):
        preds_aligned_rotated = rotate(preds_aligned, r, reshape = False, axes = (1,2))
        if im_query_augmented is None:
            im_query_augmented = jiggle_image(preds_aligned_rotated, encoder)
        else:
            im_query_augmented = np.vstack((im_query_augmented, jiggle_image(preds_aligned_rotated, encoder)))

    query_dict = {}
    nearest = []
    
    # For each class, perform a search in the corresponding index. Because we use many jiggled tiles, grab the top results from all the jiggled tiles' searches 
    # Additional search results and nearest IDs/distances are gathered as later we need to find the intersection of IDs between classes
    for class_index, c in enumerate(classes_cut):
        
        index.load_index(index_directory + '/c{}.idx'.format(c))
        
        # Perform a search for the current class
        for i in im_query_augmented:    
            temp_ids, temp_distances = index.knn_query(i[class_index], k=(1000+num_nearest*10))

            # If we've already seen an index in a search result, only update it if the distance is now lower
            for ind, value in enumerate(temp_ids[0]):
                if (value in query_dict and query_dict[value] > temp_distances[0][ind]) or (not value in query_dict):
                    query_dict[value] = temp_distances[0][ind] * weights_cut[class_index]
                    
        
        all_ids = list(query_dict.keys())
        all_distances = list(query_dict.values())
        
        # Grab the nearest ids and distances for all jiggles of the current class
        nearest_indexes = pd.Series(all_distances).nsmallest(rotations * (1000+num_nearest*10)).index.values.tolist()
        current_nearest = {}
        for i in nearest_indexes:
            current_nearest[all_ids[i]] = all_distances[i]
            
        nearest.append(current_nearest)    

    # Go through a class (the first one), checking if any of the IDs exist in other classes. If there are any IDs that exist in all classes, they are a valid search result
    # Also store the total distances for all the classes for that ID. This is used to rank the IDs we find here.
    # It doesn't matter which class we go through, so just use the first one for simplicity.
    nearest_intersection = {}
    for i in nearest[0]:

        intersection = True
        total_distance = nearest[0][i]

        for j in range(1, len(nearest)):
            if i not in nearest[j]:
                intersection = False
                break
            else:
                total_distance += nearest[j][i] * weights_cut[j]

        if intersection == True:
            nearest_intersection[i] = total_distance

    # Sort to find the smallest distances
    nearest_intersection_list = list(nearest_intersection.items())
    nearest_intersection_list.sort(key=lambda tup: tup[1])
    
    # Go from a list of tuples [(id, distance), (id, distance), ...] to a list [(id, id, ...)]
    nearest_ids = list(zip(*nearest_intersection_list))[0]
    
    result_coordinates = []
    result_images = []
    result_ids = []
    seen_coordinates = []
    
    i = 0
    j = 0
    
    if show_images:
        print('Result images')
    # Show the resultant images by grabbing the tile coordinates corresponding to their IDs
    while i < num_nearest:
        if (int(coordinates[nearest_ids[j]][0]), int(coordinates[nearest_ids[j]][1])) not in seen_coordinates:
        
            seen_coordinates.append((int(coordinates[nearest_ids[j]][0]), int(coordinates[nearest_ids[j]][1])))
            im = crop_tx_ty(int(coordinates[nearest_ids[j]][0]), 
                            int(coordinates[nearest_ids[j]][1]), 
                            int(coordinates[nearest_ids[j]][2]), 
                            int(coordinates[nearest_ids[j]][3]), 
                            zoom_level, 
                            dataset_datestr)

            result_coordinates.append((int(coordinates[nearest_ids[j]][0]), 
                                       int(coordinates[nearest_ids[j]][1]),
                                       int(coordinates[nearest_ids[j]][2]),
                                       int(coordinates[nearest_ids[j]][3])))
            result_images.append(im)
            result_ids.append(nearest_ids[j])
            
            if show_images:
                plt.imshow(im)
                plt.axis('off')
                plt.show()
            
            i += 1
            
            if i == len(nearest_ids):
                break
                
        j += 1
        
    return result_coordinates, result_images, result_ids


def search_location(encoder, index_directory, coordinates_directory, lon, lat, datestr, dataset_datestr, zoom_level, categories, category_weights = None, search_ef = 2000, num_nearest = 10, rotations = 4, index_space = 'l2', index_dim = 96, show_images = True):
    
    
    """
    Given a location in lat/long/zoom/date and categories to search with, find similar locations. Similar to search_image(), but using a real-world location instead of an image. Most arguments are passed on to search(), which is common between search_location() and search_image().
    
    Arguments:
    encoder (Keras model): Encoder from the variational autoencoder model, used to produce an encoding vector from a given image.
    index_directory (string): Folder in which the indexes are stored. One index is made per category, and they are stored in a single folder. Each index has the name "c<category>.idx".
    coordinates_directory (string): Exact location of the coordinates numpy array.
    lon (float): Longitude of query.
    lat (float): Latitude of query.
    datestr (string): Date for which to grab the query tile, in YYYY-MM-DD format.
    dataset_datestr (string): Date for which to grab the resultant similar images, in YYYY-MM-DD format.
    zoom_level (int): Zoom level over which to perform the search.
    categories (list of ints): Prediction categories to use for the search.
    category_weights (list of floats): Weights for each category, following the same order as the previous 'categories' argument. Weight = 0 means the category will not be used for the search, and weight = 1 is the default. Higher weight places more importance on that category.
    search_ef (int): Search parameter. Higher values lead to a higher quality search but longer search times. Optimal values are around 1000-5000, any higher will cause quite long searches.
    num_nearest (int): How many similar locations to return.
    rotations (int): How many rotations to perform when searching. It's a good idea to just leave this at 4 (ie 0, 90, 180, 270 degrees).
    index_space (string): Search space to use for the hnswlib search. Only 'l2', 'ip' (inner product) and 'cosine' are supported. If other spaces are necessary, consider migrating to the hnsw method in nmslib instead.
    index_dim (int): Dimensionality of the encoded predictions in the search index.
    show_images (boolean): Whether or not to show the query and result images using matplotlib.pyplot
    
    Returns:
    (result_coordinates, result_images, result_ids) tuple
    
    result_coordinates (list of tuples): A list which holds the tile coordinates (tx, ty, split_amount, crop_num) as tuples. These are the nearest neighbours, and the amount returned depends on num_nearest. The values are the same as what's used by crop_tx_ty(), but are handy if the locations need to be used elsewhere
    result_images (list of np arrays): A list of 128x128x3 numpy arrays representing the similar RGB images. The amount returned depends on num_nearest.
    result_ids (list of ints): A list which contains the IDs of each of the top N results, as they are stored in the relevant index and encoded predictions list. 
    """
    
    if (category_weights is not None) and (len(categories) != len(category_weights)):
        print('Please make sure "categories" and "category_weights" are the same length')
        return None, None, None
    
    if index_directory[-1] == '/':
        index_directory = index_directory[:-1]
    
    # Grab information about the tile and which classes exist on it
    coord = pygis.conversions.GeoCoordinates()
    tx, ty = coord.long_lat_to_tile_coordinate(lon, lat, zoom_level)
    
    url = ("https://api.nearmap.com/coverage/v2/coord/{}/{}/{}?apikey={}&until={}&limit=1"
           .format(zoom_level, tx, ty, os.environ["API_KEY"], datestr))
    response = requests.get(url).json()
    survey = response["surveys"][0]["id"]
    classes = response["surveys"][0]["resources"]["predictions"][0]["properties"]["classes"]
    classes = [int(r['id']) for r in classes]

    for c in categories:
        if c not in classes:
            print('Invalid categories. {} does not exist for this location'.format(c))
            return None, None, None
        
    classes = categories
    
    # If some of the category weights are 0, we don't need to search them at all so just exclude them from the start
    classes_cut = []
    weights_cut = []
    if category_weights != None:
        for i, c in enumerate(category_weights):
            if c != 0:
                classes_cut.append(classes[i])
                weights_cut.append(category_weights[i])
    
    # Grab the tile image and its associated predictions
    pred_req = PredictionRequester()
    tile_req = TileRequester(cache_dir = 'cache/tile_requester/') 
    preds = pred_req.get_preds_tc_with_survey(zoom_level, tx, ty, survey, classes_cut)
    scaling = 128/preds.shape[0]

    # For each of the categories align the query tile 
    preds = zoom(preds, [scaling, scaling, 1], order=0)
    preds_aligned = []
    for i in range(preds.shape[2]):  
        preds_aligned.append(rotate_tile(preds[:,:,i], padding_mode = 'constant', verbose = False))
    preds_aligned = np.stack(preds_aligned, axis = 0)
    
    tile_query = tile_req.get_tile_tc(zoom_level, tx, ty, datestr)
    tile_query = zoom(tile_query, [scaling, scaling, 1], order=0)
    
    if show_images:
        print('Query Image :')
        plt.imshow(tile_query)
        plt.axis('off')
        plt.show()
    
    return search(preds_aligned, encoder, index_directory, coordinates_directory, dataset_datestr, zoom_level, classes_cut, weights_cut, search_ef, num_nearest, rotations, index_space, index_dim, show_images)
    
    
def search_image(preds, encoder, index_directory, coordinates_directory, dataset_datestr, zoom_level, categories, category_weights = None, search_ef = 2000, num_nearest = 10, rotations = 4, index_space = 'l2', index_dim = 96, show_images = True):
    
    
    """
    Given an image and categories to search with, find similar locations. Similar to search_location(), but using an image array instead of coordinates. Most arguments are passed on to search(), which is common between search_location() and search_image().
    
    Arguments:
    preds (numpy array): A 128x128xN numpy array with values 0-1 representing the images to be used for each corresponding category. Make sure these are in the same order as the categories and category_weights!
    encoder (Keras model): Encoder from the variational autoencoder model, used to produce an encoding vector from a given image.
    index_directory (string): Folder in which the indexes are stored. One index is made per category, and they are stored in a single folder. Each index has the name "c<category>.idx".
    coordinates_directory (string): Exact location of the coordinates numpy array.
    dataset_datestr (string): Date for which to grab the resultant similar images, in YYYY-MM-DD format.
    zoom_level (int): Zoom level over which to perform the search.
    categories (list of ints): Prediction categories to use for the search.
    category_weights (list of floats): Weights for each category, following the same order as the previous 'categories' argument. Weight = 0 means the category will not be used for the search, and weight = 1 is the default. Higher weight places more importance on that category.
    search_ef (int): Search parameter. Higher values lead to a higher quality search but longer search times. Optimal values are around 1000-5000, any higher will cause quite long searches.
    num_nearest (int): How many similar locations to return.
    rotations (int): How many rotations to perform when searching. It's a good idea to just leave this at 4 (ie 0, 90, 180, 270 degrees).
    index_space (string): Search space to use for the hnswlib search. Only 'l2', 'ip' (inner product) and 'cosine' are supported. If other spaces are necessary, consider migrating to the hnsw method in nmslib instead.
    index_dim (int): Dimensionality of the encoded predictions in the search index.
    show_images (boolean): Whether or not to show the query and result images using matplotlib.pyplot
    
    Returns:
    (result_coordinates, result_images, result_ids) tuple
    
    result_coordinates (list of tuples): A list which holds the tile coordinates (tx, ty, split_amount, crop_num) as tuples. These are the nearest neighbours, and the amount returned depends on num_nearest. The values are the same as what's used by crop_tx_ty(), but are handy if the locations need to be used elsewhere
    result_images (list of np arrays): A list of 128x128x3 numpy arrays representing the similar RGB images. The amount returned depends on num_nearest.
    result_ids (list of ints): A list which contains the IDs of each of the top N results, as they are stored in the relevant index and encoded predictions list. 
    """
    
    if show_images:
        print('Query Image (first channel):')
        plt.imshow(preds[:,:,0])
        plt.axis('off')
        plt.show()
    
    if (category_weights is not None) and (len(categories) != len(category_weights)):
        print('Please make sure "categories" and "category_weights" are the same length')
        return None, None, None
    if len(categories) != preds.shape[2]:
        print('Please make sure the given preds have a 3rd dimension with the same length as "categories"')
        return None, None, None
        
    if index_directory[-1] == '/':
        index_directory = index_directory[:-1]
    
    coordinates = np.load(coordinates_directory)
    
    # If some of the category weights are 0, we don't need to search them at all so just exclude them from the start
    classes_cut = []
    weights_cut = []
    if category_weights != None:
        for i, c in enumerate(category_weights):
            if c != 0:
                classes_cut.append(categories[i])
                weights_cut.append(category_weights[i])
    
    scaling = 128/preds.shape[0]

    # For each of the categories align the query tile 
    preds = zoom(preds, [scaling, scaling, 1], order=0)
    preds_aligned = []
    for i in range(preds.shape[2]):  
        preds_aligned.append(rotate_tile(preds[:,:,i], padding_mode = 'constant', verbose = False))
    preds_aligned = np.stack(preds_aligned, axis = 0)
    
    return search(preds_aligned, encoder, index_directory, coordinates_directory, dataset_datestr, zoom_level, classes_cut, weights_cut, search_ef, num_nearest, rotations, index_space, index_dim, show_images)
    
    
def crop_rotate_encode(base_path, categories, encoder, save_path, coordinates_path, split = 5, size = 256, verbose = True):
    
    """
    Given a set of tiles fetched from the API using fetch_tiles():
        - crop them in to many overlapping tiles at the next zoom level
        - align all of those
        - encode all of those
    This effectively replaces crop_search_predictions(), rotate_search_tiles() and encode_predictions() but it does not save any of the intermediate tiles, just the resultant encodings.
    
    Arguments:
    base_path (string): The folder in which the tiles to crop/rotate/encode are stored, not including category folders
    categories (list of ints): Which categories to work on
    encoder (Keras model): Model which takes in an input 128x128 image and produces a resultant encoding vector
    save_path (string): The folder in which to save the encoded prediction numpy arrays
    coordinates_path (string): The exact location at which to save the resultant coordinates list
    split (int): How many overlapping tiles to produce for each original tile
    
    The cropping index works as follows for split = 5:
    
     _________________
    |  1  2  3  4  5  | 26
    |  6  7  8  9 10  | 27
    | 11 12 13 14 15  | 28
    | 16 17 18 19 20  | 29
    | 21 22 23 24 25  | 30
     _______________
      31 32 33 34 35    36
    
    This follows a similar structure for any value of split. For some tiles, one or more of (26,27,28,29,30), (31,32,33,34,35) or (36) may not exist as they are created by sharing with an adjacent tile (which won't exist at the edges of a dataset). 
    
    size (int): Height and width of original tiles. They should be square
    verbose (boolean): Whether or not to show a progress bar using tqdm_notebook
    
    Returns:
    None. The resultant encoded predictions and coordinates are saved according to save_path and coordinates_path.
    """
    
    
    if base_path[-1] == '/':
        base_path = base_path[:-1]
    if save_path[-1] == '/':
        save_path = save_path[:-1]
    
    os.system('mkdir -p {}'.format(save_path))
    os.system('mkdir -p {}'.format("/".join(coordinates_path.split('/')[:-1])))
    for cat in tqdm_notebook(categories, disable = not verbose, desc = "For each category"):
        file_path = base_path + '/' + str(cat)
        file_list = glob.glob(file_path + '/*')
        # Make an easy to access coordinates list for the file
        coordinates = []
        for f in file_list:
            tx = int(str(f).split('/')[-1].split('_')[0])
            ty = int(str(f).split('/')[-1].split('_')[1].split('.')[0])

            coordinates.append([tx, ty])

        crop_coordinates = []
        encodings = []
        boundaries = []

        for i in range(2*split-1):
            boundaries.append(int((size*i)/(2*split-2)))

        for f in tqdm_notebook(file_list, disable = not verbose, desc = "For each file"):

            file_extension = f.split('.')[-1]
            tx = int(str(f).split('/')[-1].split('_')[0])
            ty = int(str(f).split('/')[-1].split('_')[1].split('.')[0])

            im = np.array(PIL.Image.open(f)).astype(np.uint8)
            scaling = size/im.shape[0]
            tile = zoom(im, [scaling, scaling], order=0)

            im_right = None
            if [tx+1,ty] in coordinates:
                im_right_directory = "/".join(f.split('/')[:-1]) + '/' + str(tx+1) + '_' + str(ty) + '.' + file_extension
                im = np.array(PIL.Image.open(im_right_directory)).astype(np.uint8)
                scaling = size/im.shape[0]
                im_right = zoom(im, [scaling, scaling], order=0)

            im_bot = None
            if [tx,ty+1] in coordinates:
                im_bot_directory = "/".join(f.split('/')[:-1]) + '/' + str(tx) + '_' + str(ty+1) + '.' + file_extension
                im = np.array(PIL.Image.open(im_bot_directory)).astype(np.uint8)
                scaling = size/im.shape[0]
                im_bot = zoom(im, [scaling, scaling], order=0)

            im_bot_right = None
            if [tx+1,ty+1] in coordinates:
                im_bot_right_directory = "/".join(f.split('/')[:-1]) + '/' + str(tx+1) + '_' + str(ty+1) + '.' + file_extension
                im = np.array(PIL.Image.open(im_bot_right_directory)).astype(np.uint8)
                scaling = size/im.shape[0]
                im_bot_right = zoom(im, [scaling, scaling], order=0)

            crops = []

            # Perform crops inside the given tile
            subplot_index = 1
            for row in range(split):
                for col in range(split):
                    tile_cropped = tile[boundaries[row]:boundaries[row+split-1],boundaries[col]:boundaries[col+split-1]]

                    crops.append(tile_cropped)
                    crop_coordinates.append([tx, ty, split, subplot_index])
                    subplot_index += 1


            first_boundary = int(0.75 * size)
            second_boundary = int(0.25 * size)

            # if the right-hand tile also exists, do <split> overlapping tiles between the current one and the right-hand one
            if [tx+1,ty] in coordinates:

                subplot_index = split*split + 1
                for row in range(split):
                    tile_left = tile[boundaries[row]:boundaries[row+split-1], (first_boundary):]
                    tile_right = im_right[boundaries[row]:boundaries[row+split-1], :(second_boundary)]
                    tile_concat = np.concatenate((tile_left, tile_right), axis = 1)

                    crops.append(tile_concat)
                    crop_coordinates.append([tx, ty, split, subplot_index])

                    subplot_index += 1

            # if (ty+1,tx) exists, do some overlapping ones between (ty,tx) and (ty+1,tx)
            if [tx, ty+1] in coordinates:

                subplot_index = split*(split+1) + 1
                for col in range(split):
                    tile_top = tile[(first_boundary):, boundaries[col]:boundaries[col+split-1]]
                    tile_bot = im_bot[:(second_boundary), boundaries[col]:boundaries[col+split-1]]
                    tile_concat = np.concatenate((tile_top, tile_bot), axis = 0)

                    crops.append(tile_concat)
                    crop_coordinates.append([tx, ty, split, subplot_index])

                    subplot_index += 1

            # if (ty+1,tx+1) exists, do an overlapping one between all 4
            if [tx+1, ty+1] in coordinates:

                tile_tl = tile[first_boundary:, first_boundary:]
                tile_tr = im_right[first_boundary:, :second_boundary]
                tile_bl = im_bot[:second_boundary, first_boundary:]
                tile_br = im_bot_right[:second_boundary, :second_boundary]
                top = np.concatenate((tile_tl, tile_tr), axis = 1)
                bottom = np.concatenate((tile_bl, tile_br), axis = 1)
                tile_concat = np.concatenate((top, bottom), axis = 0)

                crops.append(tile_concat)
                crop_coordinates.append([tx, ty, split, subplot_index])

            crops_rotated = []

            for c in crops:
                crop_rotated = rotate_tile(c)
                crop_rotated = np.expand_dims(crop_rotated, axis = 2)
                crop_rotated = np.float32(crop_rotated)/255.0
                crops_rotated.append(crop_rotated)


            encoded = encoder(np.array(crops_rotated))[0].numpy()

            encodings = encodings + encoded.tolist()

        np.save('{}/c{}.npy'.format(save_path, cat), encodings)
        np.save(coordinates_path, crop_coordinates)
        

def search_brute(preds_aligned, encoder, encodings_directory, coordinates_directory, dataset_datestr, zoom_level, classes_cut, weights_cut, num_nearest, rotations, show_images):
    
    
    """
    Brute-force version of search().
    This function is used by search_location_brute() and search_image_brute() and shouldn't be called manually. 
    
    Arguments: 
    preds_aligned (numpy array): 128x128xChannels image of prediction rasters
    encoder (Keras model): Encoder from the variational autoencoder model, used to produce an encoding vector from a given image.
    coordinates_directory (string): Exact location of the coordinates numpy array.
    dataset_datestr (string): Date for which to grab the resultant similar images, in YYYY-MM-DD format.
    zoom_level (int): Zoom level over which to perform the search.
    classes_cut (list of ints): Prediction categories to use for the search, filtered such that classes corresponding to 0 weight are removed.
    weights_cut (list of floats): Weights for each category, following the same order as the previous 'categories' argument. 0 weights are removed.
    num_nearest (int): How many similar locations to return.
    rotations (int): How many rotations to perform when searching. It's a good idea to just leave this at 4 (ie 0, 90, 180, 270 degrees).
    show_images (boolean): Whether or not to show the query and result images using matplotlib.pyplot
    
    Returns:
    (result_coordinates, result_images, result_ids) tuple
    
    result_coordinates (list of tuples): A list which holds the tile coordinates (tx, ty, split_amount, crop_num) as tuples. These are the nearest neighbours, and the amount returned depends on num_nearest. The values are the same as what's used by crop_tx_ty(), but are handy if the locations need to be used elsewhere
    result_images (list of np arrays): A list of 128x128x3 numpy arrays representing the similar RGB images. The amount returned depends on num_nearest.
    result_ids (list of ints): A list which contains the IDs of each of the top N results, as they are stored in the relevant index and encoded predictions list. 
    """
    
    import nmslib
    
    if encodings_directory[-1] == '/':
        encodings_directory = encodings_directory[:-1]

    coordinates = np.load(coordinates_directory)
    
    # Jiggle the tile in order to account for some slight translation differences. 
    # Also rotate the tile 0/90/180/270 degrees
    # This means that we search many different tiles and get the best matches
    im_query_augmented = None
    for r in range(0,360,int(360/rotations)):
        preds_aligned_rotated = rotate(preds_aligned, r, reshape = False, axes = (1,2))
        if im_query_augmented is None:
            im_query_augmented = jiggle_image(preds_aligned_rotated, encoder)
        else:
            im_query_augmented = np.vstack((im_query_augmented, jiggle_image(preds_aligned_rotated, encoder)))

    query_dict = {}
    nearest = []
    
    # For each class, perform a search in the corresponding index. Because we use many jiggled tiles, grab the top results from all the jiggled tiles' searches 
    # Additional search results and nearest IDs/distances are gathered as later we need to find the intersection of IDs between classes
    
    for class_index, c in enumerate(classes_cut):
        
        index = nmslib.init(method='brute_force', space='negdotprod')
        encoded_preds = np.load(encodings_directory + '/c' + str(int(c)) + '.npy')
        index.addDataPointBatch(encoded_preds)
        index.createIndex()
        
        # Perform a search for the current class
        for i in im_query_augmented:    
            temp_ids, temp_distances = index.knnQuery(i[class_index], k=(10000 + num_nearest*10))

            # If we've already seen an index in a search result, only update it if the distance is now lower
            for ind, value in enumerate(temp_ids):
                if (value in query_dict and query_dict[value] > temp_distances[ind]) or (not value in query_dict):
                    query_dict[value] = temp_distances[ind] * weights_cut[class_index]
                    
        
        all_ids = list(query_dict.keys())
        all_distances = list(query_dict.values())
        
        # Grab the nearest ids and distances for all jiggles of the current class
        nearest_indexes = pd.Series(all_distances).nsmallest(rotations * (10000 + num_nearest*10)).index.values.tolist()
        current_nearest = {}
        for i in nearest_indexes:
            current_nearest[all_ids[i]] = all_distances[i]
        
        nearest.append(current_nearest)    

    # Go through a class (the first one), checking if any of the IDs exist in other classes. If there are any IDs that exist in all classes, they are a valid search result
    # Also store the total distances for all the classes for that ID. This is used to rank the IDs we find here.
    # It doesn't matter which class we go through, so just use the first one for simplicity.
    nearest_intersection = {}
    for i in nearest[0]:

        intersection = True
        total_distance = nearest[0][i]

        for j in range(1, len(nearest)):
            if i not in nearest[j]:
                intersection = False
                break
            else:
                total_distance += nearest[j][i] * weights_cut[j]

        if intersection == True:
            nearest_intersection[i] = total_distance

    # Sort to find the smallest distances
    nearest_intersection_list = list(nearest_intersection.items())
    nearest_intersection_list.sort(key=lambda tup: tup[1])
    
    # Go from a list of tuples [(id, distance), (id, distance), ...] to a list [(id, id, ...)]
    nearest_ids = list(zip(*nearest_intersection_list))[0]
    
    result_coordinates = []
    result_images = []
    result_ids = []
    seen_coordinates = []
    
    i = 0
    j = 0
    
    if show_images:
        print('Result images')
    # Show the resultant images by grabbing the tile coordinates corresponding to their IDs
    while i < num_nearest:
        if (int(coordinates[nearest_ids[j]][0]), int(coordinates[nearest_ids[j]][1])) not in seen_coordinates:
        
            seen_coordinates.append((int(coordinates[nearest_ids[j]][0]), int(coordinates[nearest_ids[j]][1])))
            im = crop_tx_ty(int(coordinates[nearest_ids[j]][0]), 
                            int(coordinates[nearest_ids[j]][1]), 
                            int(coordinates[nearest_ids[j]][2]), 
                            int(coordinates[nearest_ids[j]][3]), 
                            zoom_level, 
                            dataset_datestr)

            result_coordinates.append((int(coordinates[nearest_ids[j]][0]), 
                                       int(coordinates[nearest_ids[j]][1]),
                                       int(coordinates[nearest_ids[j]][2]),
                                       int(coordinates[nearest_ids[j]][3])))
            result_images.append(im)
            result_ids.append(nearest_ids[j])
            
            if show_images:
                plt.imshow(im)
                plt.axis('off')
                plt.show()
            
            i += 1
        j += 1
        if j == len(nearest_ids):
            break
        
    return result_coordinates, result_images, result_ids


def search_location_brute(encoder, encodings_directory, coordinates_directory, lon, lat, datestr, dataset_datestr, zoom_level, categories, category_weights = None, num_nearest = 10, rotations = 4, show_images = True):
    
    
    """
    Brute-force version of search_location(). Requires nmslib to be installed.
    Given a location in lat/long/zoom/date and categories to search with, find similar locations. Similar to search_image(), but using a real-world location instead of an image. Most arguments are passed on to search(), which is common between search_location() and search_image().
    
    Arguments:
    encoder (Keras model): Encoder from the variational autoencoder model, used to produce an encoding vector from a given image.
    coordinates_directory (string): Exact location of the coordinates numpy array.
    lon (float): Longitude of query.
    lat (float): Latitude of query.
    datestr (string): Date for which to grab the query tile, in YYYY-MM-DD format.
    dataset_datestr (string): Date for which to grab the resultant similar images, in YYYY-MM-DD format.
    zoom_level (int): Zoom level over which to perform the search.
    categories (list of ints): Prediction categories to use for the search.
    category_weights (list of floats): Weights for each category, following the same order as the previous 'categories' argument. Weight = 0 means the category will not be used for the search, and weight = 1 is the default. Higher weight places more importance on that category.
    num_nearest (int): How many similar locations to return.
    rotations (int): How many rotations to perform when searching. It's a good idea to just leave this at 4 (ie 0, 90, 180, 270 degrees).
    show_images (boolean): Whether or not to show the query and result images using matplotlib.pyplot
    
    Returns:
    (result_coordinates, result_images, result_ids) tuple
    
    result_coordinates (list of tuples): A list which holds the tile coordinates (tx, ty, split_amount, crop_num) as tuples. These are the nearest neighbours, and the amount returned depends on num_nearest. The values are the same as what's used by crop_tx_ty(), but are handy if the locations need to be used elsewhere
    result_images (list of np arrays): A list of 128x128x3 numpy arrays representing the similar RGB images. The amount returned depends on num_nearest.    
    result_ids (list of ints): A list which contains the IDs of each of the top N results, as they are stored in the relevant index and encoded predictions list. 
    """
        
    if (category_weights is not None) and (len(categories) != len(category_weights)):
        print('Please make sure "categories" and "category_weights" are the same length')
        return None, None, None
    
    # Grab information about the tile and which classes exist on it
    coord = pygis.conversions.GeoCoordinates()
    tx, ty = coord.long_lat_to_tile_coordinate(lon, lat, zoom_level)
    
    url = ("https://api.nearmap.com/coverage/v2/coord/{}/{}/{}?apikey={}&until={}&limit=1"
           .format(zoom_level, tx, ty, os.environ["API_KEY"], datestr))
    response = requests.get(url).json()
    survey = response["surveys"][0]["id"]
    classes = response["surveys"][0]["resources"]["predictions"][0]["properties"]["classes"]
    classes = [int(r['id']) for r in classes]

    for c in categories:
        if c not in classes:
            print('Invalid categories. {} does not exist for this location'.format(c))
            return None, None, None
        
    classes = categories
    
    # If some of the category weights are 0, we don't need to search them at all so just exclude them from the start
    classes_cut = []
    weights_cut = []
    if category_weights != None:
        for i, c in enumerate(category_weights):
            if c != 0:
                classes_cut.append(classes[i])
                weights_cut.append(category_weights[i])
    
    # Grab the tile image and its associated predictions
    pred_req = PredictionRequester()
    tile_req = TileRequester(cache_dir = 'cache/tile_requester/') 
    preds = pred_req.get_preds_tc_with_survey(zoom_level, tx, ty, survey, classes_cut)
    scaling = 128/preds.shape[0]

    # For each of the categories align the query tile 
    preds = zoom(preds, [scaling, scaling, 1], order=0)
    preds_aligned = []
    for i in range(preds.shape[2]):  
        preds_aligned.append(rotate_tile(preds[:,:,i], padding_mode = 'constant', verbose = False))
    preds_aligned = np.stack(preds_aligned, axis = 0)
    
    tile_query = tile_req.get_tile_tc(zoom_level, tx, ty, datestr)
    tile_query = zoom(tile_query, [scaling, scaling, 1], order=0)
    
    if show_images:
        print('Query Image :')
        plt.imshow(tile_query)
        plt.axis('off')
        plt.show()
    
    return search_brute(preds_aligned, encoder, encodings_directory, coordinates_directory, dataset_datestr, zoom_level, classes_cut, weights_cut, num_nearest, rotations, show_images)
    
    
def search_image_brute(preds, encoder, encodings_directory, coordinates_directory, dataset_datestr, zoom_level, categories, category_weights = None, num_nearest = 10, rotations = 4, show_images = True):
    
    
    """
    Brute-force version of search_image(). Requires nmslib to be installed.
    Given an image and categories to search with, find similar locations. Similar to search_location(), but using an image array instead of coordinates. Most arguments are passed on to search(), which is common between search_location() and search_image().
    
    Arguments:
    preds (numpy array): A 128x128xN numpy array with values 0-1 representing the images to be used for each corresponding category. Make sure these are in the same order as the categories and category_weights!
    encoder (Keras model): Encoder from the variational autoencoder model, used to produce an encoding vector from a given image.
    coordinates_directory (string): Exact location of the coordinates numpy array.
    dataset_datestr (string): Date for which to grab the resultant similar images, in YYYY-MM-DD format.
    zoom_level (int): Zoom level over which to perform the search.
    categories (list of ints): Prediction categories to use for the search.
    category_weights (list of floats): Weights for each category, following the same order as the previous 'categories' argument. Weight = 0 means the category will not be used for the search, and weight = 1 is the default. Higher weight places more importance on that category.
    num_nearest (int): How many similar locations to return.
    rotations (int): How many rotations to perform when searching. It's a good idea to just leave this at 4 (ie 0, 90, 180, 270 degrees).
    show_images (boolean): Whether or not to show the query and result images using matplotlib.pyplot
    
    Returns:
    (result_coordinates, result_images, result_ids) tuple
    
    result_coordinates (list of tuples): A list which holds the tile coordinates (tx, ty, split_amount, crop_num) as tuples. These are the nearest neighbours, and the amount returned depends on num_nearest. The values are the same as what's used by crop_tx_ty(), but are handy if the locations need to be used elsewhere
    result_images (list of np arrays): A list of 128x128x3 numpy arrays representing the similar RGB images. The amount returned depends on num_nearest.
    result_ids (list of ints): A list which contains the IDs of each of the top N results, as they are stored in the relevant index and encoded predictions list. 
    """
        
    if show_images:
        print('Query Image (first channel):')
        plt.imshow(preds[:,:,0])
        plt.axis('off')
        plt.show()
    
    if (category_weights is not None) and (len(categories) != len(category_weights)):
        print('Please make sure "categories" and "category_weights" are the same length')
        return None, None, None
    if len(categories) != preds.shape[2]:
        print('Please make sure the given preds have a 3rd dimension with the same length as "categories"')
        return None, None, None
    
    coordinates = np.load(coordinates_directory)
    
    # If some of the category weights are 0, we don't need to search them at all so just exclude them from the start
    classes_cut = []
    weights_cut = []
    if category_weights != None:
        for i, c in enumerate(category_weights):
            if c != 0:
                classes_cut.append(categories[i])
                weights_cut.append(category_weights[i])
    
    scaling = 128/preds.shape[0]

    # For each of the categories align the query tile 
    preds = zoom(preds, [scaling, scaling, 1], order=0)
    preds_aligned = []
    for i in range(preds.shape[2]):  
        preds_aligned.append(rotate_tile(preds[:,:,i], padding_mode = 'constant', verbose = False))
    preds_aligned = np.stack(preds_aligned, axis = 0)
    
    return search_brute(preds_aligned, encoder, encodings_directory, coordinates_directory, dataset_datestr, zoom_level, classes_cut, weights_cut, num_nearest, rotations, show_images)        
