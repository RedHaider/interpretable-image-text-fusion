from image_text_fusion.experiment import dataset,evaluate
def test_shape():
 i,t,y=dataset(20,2);assert i.shape==(20,10) and t.shape==(20,8) and len(y)==20
def test_metrics():assert all(0<=v<=1 for m in evaluate(3).values() for v in m.values())

