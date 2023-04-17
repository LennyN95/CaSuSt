import os
import pickle
import bz2

def createDirectories(output_dir):

    #
    if not os.path.isdir(output_dir):
        print("creating output folder:     ", output_dir)
        os.mkdir(output_dir)

    #
    for subdir in ['wip', 'raw', 'nrrd']:
        p = os.path.join(output_dir, subdir)

        if not os.path.isdir(p):
            print("creating output sub-folder: ", p)
            os.mkdir(p)


def store_compressed_pickle(fpath, data):
    with bz2.BZ2File(fpath + '.pbz2', 'wb') as f:
        pickle.dump(data, f, protocol=4)


def load_compressed_pickle(fpath):
    with bz2.BZ2File(fpath + '.pbz2', 'rb') as f:
        data = pickle.load(f)
    return data


def storeMilestones(output_dir, data, ttaugs_inv, t1subject, t2subject):
    store_compressed_pickle(os.path.join(output_dir, 'wip', 'data.prepared.pickle'), data)
    store_compressed_pickle(os.path.join(output_dir, 'wip', 'ttaugsinv.pickle'), ttaugs_inv)
    store_compressed_pickle(os.path.join(output_dir, 'wip', 't1subject.pickle'), t1subject)
    store_compressed_pickle(os.path.join(output_dir, 'wip', 't2subject.pickle'), t2subject)

def loadMilestones(output_dir):
    data = load_compressed_pickle(os.path.join(output_dir, 'wip', 'data.prepared.pickle'))
    ttaugs_inv = load_compressed_pickle(os.path.join(output_dir, 'wip', 'ttaugsinv.pickle'))
    t1subject = load_compressed_pickle(os.path.join(output_dir, 'wip', 't1subject.pickle'))
    t2subject = load_compressed_pickle(os.path.join(output_dir, 'wip', 't2subject.pickle'))
    
    print("loaded ttaugs (inv):  #", len(ttaugs_inv))
    print("loaded prepared data: #", len(data))
    print("loaded t1 subject:    #", str(t1subject))
    print("loaded t2 subject:    #", str(t2subject))

    return data, ttaugs_inv, t1subject, t2subject

####

def loadPreparedData(output_dir):
    data = load_compressed_pickle(os.path.join(output_dir, 'wip', 'data.prepared.pickle'))
    print("loaded prepared data: #", len(data))

    return data

def storePredictions(output_dir, data):
    store_compressed_pickle(os.path.join(output_dir, 'wip', 'data.predicted.pickle'), data)

def loadPredictedData(output_dir):
    data = load_compressed_pickle(os.path.join(output_dir, 'wip', 'data.predicted.pickle'))
    print("loaded predicted data: #", len(data))

    return data

#####

def storeFinalResultsForEvaluation(output_dir, roi_orig_confmaps, roi_ttaug_confmaps):
    store_compressed_pickle(os.path.join(output_dir, 'raw', 'roi_orig_confmaps.pickle'), roi_orig_confmaps)
    store_compressed_pickle(os.path.join(output_dir, 'raw', 'roi_ttaug_confmaps.pickle'), roi_ttaug_confmaps)


def loadFinalResultsForEvaluation(output_dir):
    roi_orig_confmaps = load_compressed_pickle(os.path.join(output_dir, 'raw', 'roi_orig_confmaps.pickle'))
    roi_ttaug_confmaps = load_compressed_pickle(os.path.join(output_dir, 'raw', 'roi_ttaug_confmaps.pickle'))

    return roi_orig_confmaps, roi_ttaug_confmaps