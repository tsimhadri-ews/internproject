
import kfp
from kfp import dsl

test_preprocess_op = kfp.components.load_component_from_file('test_preprocess.yaml')
test_gbc_op = kfp.components.load_component_from_file('test_gbc.yaml')
test_nb_op = kfp.components.load_component_from_file('test_nb.yaml')
test_ann_op = kfp.components.load_component_from_file('test_ann.yaml')
test_svm_op = kfp.components.load_component_from_file('test_svm.yaml')
test_logreg_op = kfp.components.load_component_from_file('test_logreg.yaml')
test_rfc_op = kfp.components.load_component_from_file('test_rfc.yaml')
test_dtc_op = kfp.components.load_component_from_file('test_dtc.yaml')

@dsl.pipeline(
    name='Malware Detection Unit Testing Pipeline',
    description='A pipeline to test the functions and training and testing results from the malware detection pipeline componenets'
)

def malware_pipeline():  #(file_path:str)
    test_preprocess = test_preprocess_op()
    test_gbc = test_gbc_op()
    test_nb = test_nb_op()
    test_ann = test_ann_op()
    test_svm = test_svm_op()
    test_logreg = test_logreg_op()
    test_rfc = test_rfc_op()
    test_dtc = test_dtc_op()
    test_preprocess.set_caching_options(False)
    test_gbc.set_caching_options(False)
    test_nb.set_caching_options(False)
    test_ann.set_caching_options(False)
    test_svm.set_caching_options(False)
    test_logreg.set_caching_options(False)
    test_rfc.set_caching_options(False)
    test_dtc.set_caching_options(False)
    
kfp.compiler.Compiler().compile(malware_pipeline, 'malware_pipeline.yaml')