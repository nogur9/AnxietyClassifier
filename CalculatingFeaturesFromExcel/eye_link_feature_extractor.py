import numpy as np
import pandas as pd
import datetime

maps_path = "../OmersData/map.xlsx"
maps = pd.read_excel(maps_path)


def get_AOI_group(row):
    map_dict = {1: "N", 2: "D"}

    CURRENT_FIX_INTEREST_AREAS, imagefile = row.split('|')
    CURRENT_FIX_INTEREST_AREAS = eval(CURRENT_FIX_INTEREST_AREAS)
    imagefile = imagefile.replace("_", " ")
    WS = "White Space"
    if CURRENT_FIX_INTEREST_AREAS == []:
        return WS
    else:
        aoi = maps[maps['SlideImage'] == imagefile]['Cell' + str(CURRENT_FIX_INTEREST_AREAS[0])]
        aoi = aoi.values[0]

        return map_dict[aoi]

class EyeLinkData:

    smi_to_eye_link_direct_tranform = {

        'Stimulus': 'imagefile',
        'Area_of_Interest': 'CURRENT_FIX_INTEREST_AREAS',
        'Fixation_Duration': 'CURRENT_FIX_DURATION',
        'Position_X': 'CURRENT_FIX_X',
        'Position_Y': 'CURRENT_FIX_Y',
        'Average_Pupil_Diameter': 'CURRENT_FIX_PUPIL',
        'Number': 'CURRENT_FIX_INDEX'}

    def __init__(self, data_path):
        data_path = data_path
        self.df = pd.read_excel(data_path)

        # removing first fixations
        self.df = self.df[self.df['CURRENT_FIX_INTEREST_AREA_PIXEL_AREA'] == 47089]


    def transform_data(self, output_path=None):
        # init
        output_path = "..\OmersData\extracted eye link data {}.xlsx".format(datetime.datetime.now().strftime('%Y-%m-%d'))
        output_df = pd.DataFrame()

        # direct transformation columns
        for col in self.smi_to_eye_link_direct_tranform:
            output_df[col] = self.df[self.smi_to_eye_link_direct_tranform[col]]

        # creating the AOI_group column
        tmp = self.df['CURRENT_FIX_INTEREST_AREAS'] + '|' + self.df['imagefile']
        output_df['AOI_Group'] = tmp.apply(get_AOI_group)

        # get subject number
        output_df['Subject'] = self.df['RECORDING_SESSION_LABEL'].str.extract(r"^([0-9]+)")

        # get Trial number
        target_col = 'identifier'
        self.df['block_num'] = pd.to_numeric(self.df['RECORDING_SESSION_LABEL'].str.extract(r"([0-9]+)$"))
        first_block_num = min(self.df['block_num'].unique())
        output_df['Trial'] = self.df['identifier'] + 30*(self.df['block_num'] - first_block_num)

        # writing the output DataFrame
        output_df.to_excel(output_path, sheet_name="fixation_data")
        demographic_df = pd.DataFrame(output_df['Subject'].unique(), columns=['Subject'])
        demographic_df.to_excel(output_path, sheet_name="demographic")

path = "..\OmersData\Book1.xlsx"
el = EyeLinkData(path)
el.transform_data()