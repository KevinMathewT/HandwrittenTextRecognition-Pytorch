import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

import pandas as pd
from ast import literal_eval
from sklearn.model_selection import StratifiedKFold

from ..utils import *
from ..utils import _clean_text
from . import det_config


def _parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []
    for child in root[0]:
        labels.append(child.attrib["text"])
    label = _clean_text(" ".join(labels))
    return label


def _get_bb_of_item(xml_file_path, level):
    if level == "full":
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        character_list = [a for a in root.iter("cmp")]
        if len(character_list) == 0:  # To account for some punctuations that have no words
            return None
        x1 = np.min([int(a.attrib['x']) for a in character_list])
        y1 = np.min([int(a.attrib['y']) for a in character_list])
        x2 = np.max([int(a.attrib['x']) + int(a.attrib['width'])
                     for a in character_list])
        y2 = np.max([int(a.attrib['y']) + int(a.attrib['height'])
                     for a in character_list])

        full_bb = [x1, y1, x2, y2]

        return [full_bb]

    if level == "line":
        bounding_boxes = []
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        line_list = [a for a in root.iter("line")]
        for line in line_list:
            character_list = [a for a in line.iter("cmp")]
            if len(character_list) == 0:  # To account for some punctuations that have no words
                return None
            x1 = np.min([int(a.attrib['x']) for a in character_list])
            y1 = np.min([int(a.attrib['y']) for a in character_list])
            x2 = np.max([int(a.attrib['x']) + int(a.attrib['width'])
                         for a in character_list])
            y2 = np.max([int(a.attrib['y']) + int(a.attrib['height'])
                         for a in character_list])

            bb = [x1, y1, x2, y2]
            bounding_boxes.append(bb)

        return bounding_boxes

    if level == "word":
        bounding_boxes = []
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        character_list = [a for a in root.iter("cmp")]
        if len(character_list) == 0:  # To account for some punctuations that have no words
            return None
        for a in character_list:
            x1 = int(a.attrib['x'])
            y1 = int(a.attrib['y'])
            x2 = int(a.attrib['x']) + int(a.attrib['width'])
            y2 = int(a.attrib['y']) + int(a.attrib['height'])

            bb = [x1, y1, x2, y2]
            bounding_boxes.append(bb)

        return bounding_boxes


def create_df():
    if os.path.isfile(config.FORMS_DF) and False:
        print(f"Loaded cached FORMS_DF from {config.FORMS_DF}")
        df = pd.read_csv(config.FORMS_DF)
        df["full_bb"] = df["full_bb"].apply(literal_eval)
        df["line_bb"] = df["line_bb"].apply(literal_eval)
    else:
        forms = glob.glob(config.FORMS_PATH + "/*/*.png") # Kaggle
        # forms = glob.glob(config.FORMS_PATH + "\*.png")  # PC
        df = pd.DataFrame(np.array(forms).reshape(-1, 1), columns=["path"])
        df["path"] = df.apply(lambda row: row.replace())
        df["image_id"] = df.apply(
            lambda row: row.path.split("/")[-1].split('.')[0], axis=1)
        df["xml"] = df.apply(lambda row: os.path.join(
            config.GENERATED_FILES_PATH, "xml") + "/" + row.image_id + ".xml", axis=1)
        df["label"] = df.apply(lambda row: _parse_xml_file(row.xml), axis=1)
        # df[["x1", "x2", "y1", "y2"]] = df.apply(lambda row: _get_bb_of_item(row.xml), axis=1)

        df["full_bb"] = df.apply(lambda row: _get_bb_of_item(row.xml, "full"), axis=1)
        df["line_bb"] = df.apply(lambda row: _get_bb_of_item(row.xml, "line"), axis=1)

        df["num_lines"] = df.apply(lambda row: len(row.line_bb), axis=1)

        df["fold"] = -1
        X = df['image_id']
        y = df["num_lines"]
        skf = StratifiedKFold(n_splits=det_config.FOLDS, shuffle=True, random_state=det_config.SEED)

        for fold, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), total=det_config.FOLDS):
            df.loc[test_index, "fold"] = fold

        df = df[['image_id', 'label', 'path', 'xml', 'full_bb', 'line_bb', 'fold']].reset_index(drop=True)

        df.to_csv(config.FORMS_DF, index=False)
        print(f"FORMS_DF cached at {config.FORMS_DF}")
    return df


if __name__ == "__main__":
    df = create_df()
    # records = df.loc[0]
    # bb = records["full_bb"]
    #
    # print(records)
    # print(bb)
    # print(bb[0])
    # print(bb[0][0])
    #
    # bb = records["line_bb"]
    # print(bb)
    # print(bb[0])
    # print(bb[0][0])

    # image_id = "a01-003"
    # image = get_img("D:\\Kevin\\Machine Learning\\IAM Dataset Full\\original\\forms\\" + image_id + ".png")
    # xml_file = image_id + ".xml"
    # bbs = _get_bb_of_item("./generated/xml/" + xml_file, "line")
    #
    # display_image_with_bb(image, bbs)
