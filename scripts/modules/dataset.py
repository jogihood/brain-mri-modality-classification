import os

class cohort:
    def __init__(self) -> None:
        self.name = "Cohort"
        self.available_modalities = ["T1", "T2", "FLAIR"]
        self.t1 = self.t1()
        self.t2 = self.t2()
        self.flair = self.flair()
        self.modalities = [self.t1, self.t2, self.flair]
        self.files = self.t1.files + self.t2.files + self.flair.files
        # self.preprocessed = self.preprocessed()

    class t1:
        def __init__(self) -> None:
            self.method = "T1"
            self.files = []
            self._basedir_ = "/nasdata4/csgradproj/data/T1"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

    class t2:
        def __init__(self) -> None:
            self.method = "T2-GRE"
            self.files = []
            self._basedir_ = "/nasdata4/csgradproj/data/T2"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

    class flair:
        def __init__(self) -> None:
            self.method = "FLAIR"
            self.files = []
            self._basedir_ = "/nasdata4/csgradproj/data/FLAIR"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

class hcp:
    def __init__(self) -> None:
        self.name = "HCP"
        self.available_modalities = ["T1", "T2"]
        self.t1 = self.t1()
        self.t2 = self.t2()
        self.modalities = [self.t1, self.t2]
        self.files = self.t1.files + self.t2.files

    class t1:
        def __init__(self) -> None:
            self.method = "MPRAGE"
            self.files = []
            self._basedir_ = "/nasdata4/csgradproj/HCP/T1"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_:
                self.files.append(os.path.join(self._basedir_, f))

    class t2:
        def __init__(self) -> None:
            self.method = "T2-SPC"
            self.files = []
            self._basedir_ = "/nasdata4/csgradproj/HCP/T2"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_:
                self.files.append(os.path.join(self._basedir_, f))

class adni1:
    def __init__(self) -> None:
        self.name = "ADNI1"
        self.available_modalities = ["T1", "T2"]
        self.t1 = self.t1()
        self.t2 = self.t2()
        self.modalities = [self.t1, self.t2]
        self.files = self.t1.files + self.t2.files

    class t1:
        def __init__(self) -> None:
            self.method = "MPRAGE"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/ADNI/ADNI1/T1/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, "NIFTI"]
                p = os.path.join(*l)
                self.files.append(os.path.join(p, os.listdir(p)[0]))

    class t2:
        def __init__(self) -> None:
            self.method = "T2-FSE"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/ADNI/ADNI1/T2/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, "NIFTI"]
                p = os.path.join(*l)
                self.files.append(os.path.join(p, os.listdir(p)[0]))

class adni2:
    def __init__(self) -> None:
        self.name = "ADNI2"
        self.available_modalities = ["T1", "FLAIR"]
        self.t1 = self.t1()
        self.flair = self.flair()
        self.modalities = [self.t1, self.flair]
        self.files = self.t1.files + self.flair.files

    class t1:
        def __init__(self) -> None:
            self.method = "MPRAGE"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/ADNI/ADNI2/T1/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, "NIFTI"]
                p = os.path.join(*l)
                self.files.append(os.path.join(p, os.listdir(p)[0]))

    class flair:
        def __init__(self) -> None:
            self.method = "FLAIR (Axial)"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/ADNI/ADNI2/FLAIR/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, "NIFTI"]
                p = os.path.join(*l)
                self.files.append(os.path.join(p, os.listdir(p)[0]))

class adni3:
    def __init__(self) -> None:
        self.name = "ADNI3"
        self.available_modalities = ["T1", "FLAIR"]
        self.t1 = self.t1()
        self.flair = self.flair()
        self.modalities = [self.t1, self.flair]
        self.files = self.t1.files + self.flair.files

    class t1:
        def __init__(self) -> None:
            self.method = "MPRAGE (Sagittal)"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/ADNI/ADNI3/T1/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, "NIFTI"]
                p = os.path.join(*l)
                self.files.append(os.path.join(p, os.listdir(p)[0]))

    class flair:
        def __init__(self) -> None:
            self.method = "FLAIR (Sagittal)"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/ADNI/ADNI3/FLAIR/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, "NIFTI"]
                p = os.path.join(*l)
                self.files.append(os.path.join(p, os.listdir(p)[0]))

class adnigo:
    def __init__(self) -> None:
        self.name = "ADNIGO"
        self.available_modalities = ["T1", "FLAIR"]
        self.t1 = self.t1()
        self.flair = self.flair()
        self.modalities = [self.t1, self.flair]
        self.files = self.t1.files + self.flair.files

    class t1:
        def __init__(self) -> None:
            self.method = "MPRAGE"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/ADNI/ADNIGO/T1/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, "NIFTI"]
                p = os.path.join(*l)
                self.files.append(os.path.join(p, os.listdir(p)[0]))

    class flair:
        def __init__(self) -> None:
            self.method = "FLAIR (Axial)"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/ADNI/ADNIGO/FLAIR/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, "NIFTI"]
                p = os.path.join(*l)
                self.files.append(os.path.join(p, os.listdir(p)[0]))

class camcan:
    def __init__(self) -> None:
        self.name = "CamCAN"
        self.available_modalities = ["T1", "T2"]
        self.t1 = self.t1()
        self.t2 = self.t2()
        self.modalities = [self.t1, self.t2]
        self.files = self.t1.files + self.t2.files

    class t1:# dimension wrong
        def __init__(self) -> None:
            self.method = "T1"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/CAMCAN/T1/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                l = [self._basedir_, s, f"{s}_T1w.nii.gz"]
                self.files.append(os.path.join(*l))

    class t2:# dimension wrong
        def __init__(self) -> None:
            self.method = "T2"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/CAMCAN/T2/raw"
            self._subject_list_ = os.listdir(self._basedir_)
            for s in self._subject_list_:
                if ".sh" in s : continue
                l = [self._basedir_, s, f"{s}_T2w.nii.gz"]
                self.files.append(os.path.join(*l))

class ixi:
    def __init__(self) -> None:
        self.name = "IXI"
        self.available_modalities = ["T1", "T2"]
        self.t1 = self.t1()
        self.t2 = self.t2()
        self.modalities = [self.t1, self.t2]
        self.files = self.t1.files + self.t2.files

    class t1:# dimension wrong
        def __init__(self) -> None:
            self.method = "T1"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/IXI/T1/raw"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

    class t2:
        def __init__(self) -> None:
            self.method = "T2"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/IXI/T2/raw"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

class kirby21:
    def __init__(self) -> None:
        self.name = "Kirby21"
        self.available_modalities = ["T1", "T2", "FLAIR"]
        self.t1 = self.t1()
        self.t2 = self.t2()
        self.flair = self.flair()
        self.modalities = [self.t1, self.t2, self.flair]
        self.files = self.t1.files + self.t2.files + self.flair.files

    class t1:
        def __init__(self) -> None:
            self.method = "MPRAGE"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/Kirby-21/T1/raw"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ :
                if ".par" not in f : self.files.append(os.path.join(self._basedir_, f))

    class t2:
        def __init__(self) -> None:
            self.method = "T2"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/Kirby-21/T2/raw"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ :
                if ".par" not in f : self.files.append(os.path.join(self._basedir_, f))

    class flair:
        def __init__(self) -> None:
            self.method = "FLAIR"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/Kirby-21/FLAIR/raw"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ :
                if ".par" not in f : self.files.append(os.path.join(self._basedir_, f))

class miccai2017:
    def __init__(self) -> None:
        self.name = "MICCAI2017"
        self.available_modalities = ["T1", "FLAIR"]
        self.t1 = self.t1()
        self.flair = self.flair()
        self.modalities = [self.t1, self.flair]
        self.files = self.t1.files + self.flair.files

    class t1:
        def __init__(self) -> None:
            self.method = "T1"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/MICCAI_2017/T1/raw"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ :
                if "mask" not in f : self.files.append(os.path.join(self._basedir_, f))

    class flair:
        def __init__(self) -> None:
            self.method = "FLAIR"
            self.files = []
            self._basedir_ = "/nasdata4/CNAlab_data/open_data/MICCAI_2017/FLAIR/raw"
            self._files_ = os.listdir(self._basedir_)
            for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

# class miccai2018:
#     def __init__(self) -> None:
#         self.available_modalities = ["T1", "T2", "FLAIR"]
#         self.t1 = self.t1()
#         self.t2 = self.t2()
#         self.flair = self.flair()
#         self.files = self.t1.files + self.t2.files + self.flair.files

#     class t1:
#         def __init__(self) -> None:
#             self.method = "T1"
#             self.files = []
#             self._basedir_ = "/nasdata4/CNAlab_data/open_data/MICCAI_2018/T1/raw/MICCAI_BraTS_2018_Data_Training/High_Grade_Gliomas"
#             self._files_ = os.listdir(self._basedir_)
#             for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))
#             self._basedir_ = "/nasdata4/CNAlab_data/open_data/MICCAI_2018/T1/raw/MICCAI_BraTS_2018_Data_Training/Low_Grade_Gliomas"
#             self._files_ = os.listdir(self._basedir_)
#             for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

#     class t2:
#         def __init__(self) -> None:
#             self.method = "T2"
#             self.files = []
#             self._basedir_ = "/nasdata4/CNAlab_data/open_data/MICCAI_2018/T2/raw/MICCAI_BraTS_2018_Data_Training/High_Grade_Gliomas"
#             self._files_ = os.listdir(self._basedir_)
#             for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))
#             self._basedir_ = "/nasdata4/CNAlab_data/open_data/MICCAI_2018/T2/raw/MICCAI_BraTS_2018_Data_Training/Low_Grade_Gliomas"
#             self._files_ = os.listdir(self._basedir_)
#             for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

#     class flair:
#         def __init__(self) -> None:
#             self.method = "FLAIR"
#             self.files = []
#             self._basedir_ = "/nasdata4/CNAlab_data/open_data/MICCAI_2018/FLAIR/raw/MICCAI_BraTS_2018_Data_Training/High_Grade_Gliomas"
#             self._files_ = os.listdir(self._basedir_)
#             for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))
#             self._basedir_ = "/nasdata4/CNAlab_data/open_data/MICCAI_2018/FLAIR/raw/MICCAI_BraTS_2018_Data_Training/Low_Grade_Gliomas"
#             self._files_ = os.listdir(self._basedir_)
#             for f in self._files_ : self.files.append(os.path.join(self._basedir_, f))

class oasis1:
    def __init__(self) -> None:
        self.available_modalities = ["T1"]
        self.t1 = self.t1()
        self.modalities = [self.t1]
        self.files = self.t1.files

    class t1:
        def __init__(self) -> None:
            self.method = "MPRAGE"
            self.files = []
            self._basedir_ = "/nasdata4/csgradproj/OASIS-1/NIfTI"
            self._subjects_ = os.listdir(self._basedir_)
            for s in self._subjects_ :
                sd = os.path.join(self._basedir_, s)
                for f in os.listdir(sd):
                    self.files.append(os.path.join(sd, f))

class oasis3:
    def __init__(self) -> None:
        self.available_modalities = ["T1", "T2", "FLAIR"]
        self.t1 = self.t1()
        self.t2 = self.t2()
        self.flair = self.flair()
        self.modalities = [self.t1, self.t2, self.flair]
        self._basedir_ = "/nasdata4/csgradproj/OASIS-3"
        for mrid in os.listdir(self._basedir_):
            mrid = os.path.join(self._basedir_, mrid)
            for anat in os.listdir(mrid):
                anat = os.path.join(mrid, anat)
                if os.path.isfile(anat): continue
                for f in os.listdir(anat):
                    if "nii.gz" in f and (not "hippocampus" in f):
                        f = os.path.join(anat, f)
                        if "T1" in f: self.t1.files.append(f)
                        elif "T2" in f: self.t2.files.append(f)
                        elif "FLAIR" in f: self.flair.files.append(f)
        self.files = self.t1.files + self.t2.files + self.flair.files

    class t1:
        def __init__(self) -> None:
            self.method = "T1"
            self.files = []

    class t2:
        def __init__(self) -> None:
            self.method = "T2"
            self.files = []

    class flair:
        def __init__(self) -> None:
            self.method = "FLAIR"
            self.files = []
            
cohort      = cohort()
hcp         = hcp()
adni1       = adni1()
adni2       = adni2()
adni3       = adni3()
adnigo      = adnigo()
camcan      = camcan()
ixi         = ixi()
kirby21     = kirby21()
miccai2017  = miccai2017()
oasis1      = oasis1()
oasis3      = oasis3()
# miccai2018  = miccai2018()
files       = cohort.files + hcp.files + adni1.files + adni2.files + adni3.files + adnigo.files + camcan.files + ixi.files + kirby21.files + miccai2017.files

base_2d     = "/nasdata4/csgradproj/2d_data/224"
images_2d   = []
for _f_ in os.listdir(base_2d): images_2d.append(os.path.join(base_2d, _f_))

mod_count = [0,0,0]
for _f_ in files:
    if "T1" in _f_: mod_count[0] += 1
    elif "T2" in _f_: mod_count[1] += 1
    elif "FLAIR" in _f_: mod_count[2] += 1