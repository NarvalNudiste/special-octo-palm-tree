import numpy as np

def labelize(subjects):
    for i in range(len(subjects)):
        to_remove = ["binary_output", "id", "multiclass_ouput", "temp", "bvp", "hr", "eda", "tags", "timestamps"]
        attr = [e for e in dir(subjects[i]) if not e.startswith('__') and not callable(getattr(subjects[i],e))]
        for e in to_remove:
            attr.remove(e)
        '''
        for i in range(len(attr)):
            subjects[attr[i]]= np.full(shape, subjects[attr[i]])
            print(attr[i])
            print(subjects.attr[i])
            print(subjects.attr[i].shape)
        '''
        for i in range(len(subjects)):
            shape = subjects[i].bvp.shape[0]
            subjects[i].overall_health = np.full(shape, subjects[i].overall_health)
            subjects[i].overall_stress = np.full(shape, subjects[i].overall_stress)
            subjects[i].energetic = np.full(shape, subjects[i].energetic)
            subjects[i].sleep_quality_past_24h = np.full(shape, subjects[i].sleep_quality_past_24h)
            subjects[i].sleep_quality_past_month = np.full(shape, subjects[i].sleep_quality_past_month)
            subjects[i].stressed_past_24h = np.full(shape, subjects[i].stressed_past_24h)
