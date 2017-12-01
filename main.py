__author__ = 'Jakob Abesser'

from importer import Importer



if __name__ == '__main__':

    dir_root = '/Volumes/MINI/guitar_pro'
    dir_data = '/Volumes/MINI/guitar_pro_data'

    importer = Importer(dir_root=dir_root,
                        dir_data=dir_data)
    importer.run()


    print('done :)')