from sklearn.svm import SVC


lin_svm_grid_search = [
    {'svm_linear__kernel': ['linear'], 'svm_linear__gamma': ['auto'],
     'svm_linear__C': [0.1, 1, 10], 'svm_linear__class_weight': ['balanced']}
]

CLASSIFIER_PARAMS = {
    'svm_linear': (('svm_linear', SVC()), lin_svm_grid_search),
}
