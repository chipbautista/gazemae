from scipy import io


class Biometrics():
    def get_xy(self, data):
        _bool = data.corpus.apply(
            lambda x: x in ['EMVIC2014', 'Cerf2007-FIFA', 'ETRA2019'])
        data = data[_bool]
        data = data[~data['subj'].str.contains('test')]
        return data.z, data.subj


class Biometrics_EMVIC():
    def get_xy(self, data):
        data = data[data.corpus == 'EMVIC2014']
        data = data[~data.subj.str.contains('test')]
        return data.z, data.subj

    def get_test(self, data):
        data = data[data.corpus == 'EMVIC2014']
        data = data[data.subj.str.contains('test')]
        data.subj = data.subj.apply(lambda x: x.replace('test-', ''))
        return data.z, data.subj


class Biometrics_FIFA():
    def get_xy(self, data):
        data = data[data.corpus == 'Cerf2007-FIFA']
        return data.z, data.subj


class Biometrics_ETRA():
    def get_xy(self, data):
        data = data[data.corpus == 'ETRA2019']
        return data.z, data.subj


class Biometrics_MIT_LR():
    def get_xy(self, data):
        _bool = data.corpus.apply(lambda x: x == 'MIT-LowRes')
        data = data[_bool]
        return data.z, data.subj


class ETRAStimuli():
    def get_xy(self, data):
        data = data[data.corpus == 'ETRA2019']
        # y = {'Blank': 0, 'Natural': 1, 'Puzzle': 2, 'Waldo': 3}
        # stim_type = data.task.apply(lambda t: y[t.split('_')[0]])
        stim_type = data.task.apply(lambda t: t.split('_')[0])
        return data.z, stim_type


class AgeGroupBinary():
    def get_xy(self, data):
        def bin(s):
            age = ages[s]
            if age in range(18, 23):
                return '18-22'
            return '23-35'

        data = data[data.corpus == 'Cerf2007-FIFA']
        subjects = io.loadmat(
            '../data/Cerf2007-FIFA/general', squeeze_me=True)['subject']
        ages = {s[4]: s[2] for s in subjects}
        return data.z, data['subj'].apply(lambda x: bin(x))


class GenderBinary():
    def get_xy(self, data):
        data = data[data.corpus == 'Cerf2007-FIFA']
        subjects = io.loadmat(
            '../data/Cerf2007-FIFA/general', squeeze_me=True)['subject']
        legend = {1: 'male', 0: 'female'}
        subj_sexes = {s[4]: legend[s[0]] for s in subjects}
        return data.z, data['subj'].apply(lambda x: subj_sexes[x])


TASKS = {
    'biometrics-emvic': Biometrics_EMVIC,
    'biometrics-emvic-test': Biometrics_EMVIC,
    'biometrics-all': Biometrics,
    'stimuli-etra': ETRAStimuli,
    'age-group': AgeGroupBinary,
    'gender': GenderBinary
}
