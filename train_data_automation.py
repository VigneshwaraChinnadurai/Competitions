# organize imports
import os
import glob
import datetime

# get the input and output path
input_path  = "C:\\Users\\91986\\Desktop\\Deep Learning\\Practice\\Hacker Earth\\Garden Nerd\\data\\train_bf_automation"
output_path = "C:\\Users\\91986\\Desktop\\Deep Learning\\Practice\\Hacker Earth\\Garden Nerd\\data\\train"

# get the class label limit
class_limit = 102

# take all the images from the dataset
image_paths = glob.glob(input_path + "\\*.jpg")

# variables to keep track
label = 0
i = 0
j = 80

# flower17 class names
class_names = ["Alpine sea holly", "Anthurium", "Artichoke", "Azalea", "Ball Moss",
			   "Balloon Flower", "Barbeton Daisy","Bearded Iris", "Bee Balm", "Bird of paradise", 
               "Bishop of llandaff", "Blackberry Lily", "Black-eyed Susan", "Blanket flower", "Bolero deep blue", 
               "Bougainvillea","Bromelia", "Buttercup","Californian Poppy","Camellia",
               "Canna Lily","Canterbury Bells","Cape Flower","Carnation","Cautleya Spicata",
               "Clematis","Colt's Foot","Columbine","Common Dandelion","Corn poppy",
               "Cyclamen","Daffodil","Desert-rose","English Marigold","Fire Lily",
               "Foxglove","Frangipani","Fritillary","Garden Phlox","Gaura",
               "Gazania","Geranium","Giant white arum lily","Globe Thistle","Globe-flower",
               "Grape Hyacinth","Great Masterwort","Hard-leaved pocket orchid","Hibiscus","Hippeastrum",
               "Japanese Anemone","King Protea","Lenten Rose","Lotus","Love in the mist",
               "Magnolia","Mallow","Marigold","Mexican Aster","Mexican Petunia",
               "Monkshood","Moon Orchid","Morning Glory","Orange Dahlia","Osteospermum",
               "Oxeye Daisy","Passion Flower","Pelargonium","Peruvian Lily","Petunia",
               "Pincushion flower","Pink Primrose","Pink-yellow Dahlia","Poinsettia","Primula",
               "Prince of wales feathers","Purple Coneflower","Red Ginger","Rose","Ruby-lipped Cattleya",
               "Siam Tulip","Silverbush","Snapdragon","Spear Thistle","Spring Crocus",
               "Stemless Gentian","Sunflower","Sweet pea","Sweet William","Sword Lily",
               "Thorn Apple","Tiger Lily","Toad Lily","Tree Mallow","Tree Poppy",
               "Trumpet Creeper","Wallflower","Water Lily","Watercress","Wild Pansy",
               "Windflower","Yellow Iris"
               ]

# change the current working directory
os.chdir(output_path)

# loop over the class labels
for x in range(1, class_limit+1):
	# create a folder for that class
	os.system("mkdir " + class_names[label])
	# get the current path
	cur_path = output_path + "\\" + class_names[label] + "\\"
	# loop over the images in the dataset
	for image_path in image_paths[i:j]:
		original_path = image_path
		image_path = image_path.split("\\")
		image_path = image_path[len(image_path)-1]
		os.system("copy " + original_path + " " + cur_path + image_path)
	i += 80
	j += 80
	label += 1
    
    