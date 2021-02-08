import numpy as np
import pickle
import prepare_data_for_prediction



def separate_leaf_wood(in_file, clf, leaf_out_file, wood_out_file):

    data = np.loadtxt(in_file)
    data_with_features = prepare_data_for_prediction.get_all_features(data, in_file)
    data_feat = data_with_features[:,3:]

    data_feat[np.isnan(data_feat)] = 0

    
    y_pred = clf.predict(data_feat)
    all_pts = np.column_stack((data[:,0:3], y_pred))
    wood_pts = all_pts[all_pts[:, -1] == 1]
    leaf_pts = all_pts[all_pts[:, -1] == 0]
    np.savetxt(wood_out_file, wood_pts, fmt='%1.3f')
    np.savetxt(leaf_out_file, leaf_pts, fmt='%1.3f')

    



model_file = 'leaf_wood_RF_final_model.sav' #download the model from the following link: https://www.dropbox.com/s/dpe8hzxorufv7qt/leaf_vs_wood_clf_model.sav?dl=0
in_folder = 'C:/Users/FieldLaptop/Dropbox/PhD/Results/BCI_2019/50ha_plot/Final_extracted_trees/Pointclouds/Failed_pt_cloud/' #path of the folder with all the input files

clf = pickle.load(open(model_file, 'rb'))

file_names = glob.glob(in_folder + '*.txt')


for i in range(len(file_names)):
    in_file = file_names[i]
    leaf_out_file = in_file + '_leaf.txt'
    wood_out_file = in_file + '_wood.txt'
    separate_leaf_wood(in_file, clf, leaf_out_file, wood_out_file)


