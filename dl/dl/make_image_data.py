import os
import pandas as pd
import shutil
from torchvision.io import read_image
from torchvision.transforms.functional import get_image_size, get_image_num_channels
from sklearn.model_selection import train_test_split


def create_df_from_all_images(root):
    sources = []
    labels = []
    species = []
    
    for folder in os.listdir(root):
        splitted_folder = folder.split('___')
        specie = splitted_folder[0]
        label_ = splitted_folder[1]
        
        if label_ == 'healthy':
            label = 'healthy'
        else:
            label = 'diseased'
        
        folder_path = root + "/" + folder
        for image in os.listdir(folder_path):
            source = str(f"{root}/{specie}___{label_}/{image}")
            sources.append(source)
            labels.append(label)
            species.append(specie)
            
    df = pd.DataFrame(
        {
            "source" : sources,
            "label" : labels,
            "specie": species
        }
    )
    
    return df

def drop_incorrect_images_from_metadata(df:pd.DataFrame, expected_img_size:list, expected_nr_channels:int) -> pd.DataFrame:
    expected_image_size = expected_img_size
    expected_nr_channels = expected_nr_channels
    
    for i in range(len(df)):
        image = read_image(df.loc[i, "source"])
        actual_image_size = get_image_size(image)
        actual_channels = get_image_num_channels(image)
    
        if expected_image_size != actual_image_size or expected_nr_channels != actual_channels:
            df = df.drop(i)
    
    return df.reset_index(drop=True)


# Function to remove images for specified species
def remove_images_for_species(df, species, count):
    df = df.copy()
    species_rows = df[(df['label'] == 'diseased') & (df['specie'] == species)]
    rows_to_remove = species_rows.sample(n=int(count), random_state=42)
    df = df.drop(rows_to_remove.index) 
    return df.reset_index(drop=True)
 

def remove_removals(df, removals):
    df = df.copy()
    for species, count in removals.items():
        df = remove_images_for_species(df, species, count)
    return df

def select_chosen_species(df,species):
    df = df.copy()

    df = df.loc[df['specie'].isin(species)]
    
    return df.reset_index(drop = True)


def split_image_data(df):
    df = df.copy()
    X = df.loc[:, ["source", "specie"]]
    y = df.drop(["source", "specie"], axis=1)

    # train en test set maken - 0.2 test - stratify zorgt ervoor dat de verhouding DS en HL hetzelfde is in training en test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # validatie set aanmaken - 0.5 van test set 
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # join de labels met metadata
    all_data = X.merge(y, left_index=True, right_index=True).reset_index(drop=True)
    train_data = X_train.merge(y_train, left_index=True, right_index=True).reset_index(drop=True)
    test_data = X_test.merge(y_test, left_index=True, right_index=True).reset_index(drop=True)
    val_data = X_val.merge(y_val, left_index=True, right_index=True).reset_index(drop=True)
    
    return all_data, train_data, test_data, val_data

def add_split_column(df, split):
    df = df.copy()
    df["split"] = split
    
    return df

def add_destination_column(df):
    df = df.copy()
    df['destination'] = "data" +'/'+ df["split"] +'/'+ df["label"] +'/'+ df["source"].str.split("/").str[-1]
    
    return df

def copy_from_source_to_dest(df):
    for index, row in df.iterrows():
        src = row["source"]
        dst = row["destination"]
        shutil.copyfile(src, dst)
        


# Remove the specified number of images for each species based on predefined amounts (see visualization.ipynb)
removals = {
    'Grape': 2925.307529,
    'Peach': 1846.504917,
    'Potato': 1607.753520,
    'Tomato': 13319.434034
}

species = ["Apple", "Cherry", "Corn", "Grape", "Peach", "Pepper", "Potato", "Strawberry", "Tomato"]

if __name__ == '__main__':

    os.makedirs("data/train/diseased")
    os.makedirs("data/train/healthy")
    os.makedirs("data/validation/diseased")
    os.makedirs("data/validation/healthy")
    os.makedirs("data/test/diseased")
    os.makedirs("data/test/healthy")

    
    df = create_df_from_all_images('data/raw/color')
    
    df = select_chosen_species(df, species)
    
    df = drop_incorrect_images_from_metadata(df, [256,256], 3)
    
    df = remove_removals(df, removals)
    all_data, train, test, val = split_image_data(df)
    
    train = add_split_column(train, "train")
    val = add_split_column(val, "validation")
    test = add_split_column(test, "test")
    
    
    train = add_destination_column(train)
    val = add_destination_column(val)
    test = add_destination_column(test)
    
    for split in [train, val, test]:
        copy_from_source_to_dest(split)




  