import numpy as np


def decoupe_im(results, im):
    """ Store all face in a liste"""
    all_face = []
    box_coor = []
    for i in range(len(results)):
        x, y, w, h = results[i]['box']  # recupre les bbox
        x1, y1 = abs(x), abs(y)
        x2, y2 = x1 + w, y1 + h
        face = im[y1:y2, x1:x2]
        all_face.append(face)
        box_coor.append((x, y, w, h))
    return all_face, box_coor


def extract_face(image):
    """Function to detect all face in image"""
    # im = cv2.imread(image) #read image
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert image to RGB
    results = detectors.detect_faces(im)  # detect all face in image
    all_face, box_coor = decoupe_im(results, im)  # store all image in liste
    all_face = [cv2.resize(all_face[i], (160, 160)) for i in range(len(all_face))]  # resize all image in liste
    all_face = np.array(all_face)
    return all_face, box_coor


def extract_face_path(image):
    """Function to detect all face in image"""
    im = cv2.imread(image)  # read image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # convert image to RGB
    results = detectors.detect_faces(im)  # detect all face in image
    all_face, box_coor = decoupe_im(results, im)  # store all image in liste
    all_face = [cv2.resize(all_face[i], (160, 160)) for i in range(len(all_face))]  # resize all image in liste
    all_face = np.array(all_face)
    return all_face, box_coor


def chargement_faces(path):
    faces = []
    # print(listdir(path))
    for x in listdir(path):
        # print("*** " , x)
        path1 = path + x
        face, _ = extract_face_path(path1)
        faces.append(face)
    return faces


def load_dataset(directory):
    """Directory hatrany am train na val"""
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        print(path)
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = chargement_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


def piplene_face(model, face_pixels):
    # scale pixel values
    # face_pixels = face_pixels.reshape(160,160,3)

    face_pixels = face_pixels.astype('float32')

    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    print(samples.shape)
    yhat = model.predict(samples)

    return yhat[0]


def piplene_face_train(model, face_pixels):
    # scale pixel values
    # face_pixels = face_pixels.reshape(160,160,3)

    face_pixels = face_pixels.astype('float32')

    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    print(samples[0].shape)
    yhat = model.predict(samples[0])


def piplene_face_test(model, face_pixels):
    # scale pixel values
    # face_pixels = face_pixels.reshape(160,160,3)

    face_pixels = face_pixels.astype('float32')

    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    print(samples.shape)
    yhat = model.predict(samples)

    return yhat[0]


def remove_values_by_indices(input_list, indices_to_remove):
    # Create a new list to store the values after removal
    result_list = []

    # Iterate through the elements in the input list and their indices
    for index, value in enumerate(input_list):
        # Check if the current index should be removed
        if index not in indices_to_remove:
            # If not, add the value to the result list
            result_list.append(value)

    return result_list


def standardize_data(datas):
    data_standardize = []
    for data in datas:
        m, st = data.mean(), data.std()
        data = (data - m) / st
        data_standardize.append(data)

    return data_standardize


def data_processing(directory_train):
    index_to_remove, x_train_fic = [], []
    x_train, y_train = load_dataset(directory_train)

    for i in range(len(x_train)):
        if x_train[i].shape[0] != 0:
            x_train_fic.append(x_train[i][0])
        else:
            index_to_remove.append(i)

    y_train = remove_values_by_indices(y_train, index_to_remove)

    x_train = standardize_data(x_train_fic)

    return y_train, x_train
