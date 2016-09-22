import os
import dicom
import fnmatch
import re
import os.path
from dicom.filereader import InvalidDicomError
import logging

logger = logging.getLogger()


def get_img_mask_dict(img_dir, mask_dir):
    logger.info("processing {0}".format(img_dir))
    img_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(
        img_dir) for f in fnmatch.filter(files, "IMG*")]
    img_dict = {}
    for img in img_list:
        try:
            f = dicom.read_file(img)
        except IOError:
            print 'No such file'
        except InvalidDicomError:
            print 'Invalid Dicom file {0}'.format(img)
        patientID = f.PatientID
        if hasattr(f, 'SeriesNumber'):
            seriesNumber = f.SeriesNumber
        else:
            seriesNumber = 'ELSE'
        if patientID not in img_dict:
            img_dict[patientID] = {}
            img_dict[patientID]['age'] = f.PatientsAge
            img_dict[patientID]['sex'] = f.PatientsSex
            img_dict[patientID]['img_series'] = {}
        if seriesNumber not in img_dict[patientID]['img_series']:
            img_dict[patientID]['img_series'][seriesNumber] = {}
            img_dict[patientID]['img_series'][seriesNumber][
                'path'] = re.search(r'(.*SRS\d{1,})(/IMG\d{1,})', img).group(1)
            img_dict[patientID]['img_series'][seriesNumber]['imgs'] = []
            img_dict[patientID]['img_series'][seriesNumber]['masks'] = []
            img_dict[patientID]['img_series'][seriesNumber]['img_number'] = 0
            img_dict[patientID]['img_series'][seriesNumber]['mask_number'] = 0

        img_dict[patientID]['img_series'][seriesNumber]['imgs'].append(img)
        img_dict[patientID]['img_series'][seriesNumber]['img_number'] += 1
        path = mask_dir + '/' + re.search(r'(.*DICOMDAT/)(SDY.*)', img_dict[patientID][
            'img_series'][seriesNumber]['path']).group(2)
        mask_name = re.search(
            r'(.*SRS\d{1,}/)(IMG\d{1,})', img).group(2) + '-mask.tif'
        mask_path = os.path.join(path, mask_name)
        # print mask_path
        if os.path.isfile(mask_path):
            img_dict[patientID]['img_series'][
                seriesNumber]['masks'].append(mask_path)
            img_dict[patientID]['img_series'][seriesNumber]['mask_number'] += 1
        else:
            img_dict[patientID]['img_series'][
                seriesNumber]['masks'].append(None)

    img_n = 0
    case_n = 0
    for p in img_dict.keys():
        print '#' * 30
        print 'patient id : ' + p
        for s in img_dict[p]['img_series'].keys():
            img_n += img_dict[p]['img_series'][s]['img_number']
            case_n += img_dict[p]['img_series'][s]['mask_number']
            print 'series number: ' + str(s) + ' | dir_path: ' + img_dict[p]['img_series'][s]['path'] + " | number of images: " + str(img_dict[p]['img_series'][s]['img_number']) + " | number of masks: " + str(img_dict[p]['img_series'][s]['mask_number'])
    print 'Total number of patients: ' + str(len(img_dict.keys()))
    print 'Total number of images: ' + str(img_n)
    print 'Total number of cases: ' + str(case_n)

    return img_dict
