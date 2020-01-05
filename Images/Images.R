setwd("./Données du TP 10-20191217/images_train")
img_dir <- "./cat"

library(EBImage)
library(pbapply)

width <- 32
height <- 32

read_images <- function(img_dir, width, height, tag) {
  img_size <- width*height
  # récupérer la liste de noms du image
  images_names <- list.files(img_dir)
  res <- NULL
  
  print(paste("Start processing", length(images_names), "images"))
  
  ####### create feature list #########
  feature_list <- pblapply(images_names, function(imgname) {
    img <- readImage(file.path(img_dir,imgname))
    # resize image
    img_resized <- resize(img, w = width, h = height)
    # grayscale
    img_gray <- channel(img_resized, "gray")
    # get the image matrix
    img_matrix <- img_gray@.Data
    # turn the image matrix to image vector
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  }) 
  
  ## bind the list of vector into matrix
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  ## Set names
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  
  ## Add labels
  feature_matrix <- add_label(feature_matrix,label = tag)

  return(feature_matrix)
}

add_label <- function(raw_feature_matrix, label){
  raw_feature_matrix <- cbind(label = label, raw_feature_matrix)
  return(raw_feature_matrix)
}

cats_data <- read_images("./cat", width, height, "cat")
cars_data <- read_images("./car", width, height, "car")
flowers_data <- read_images("./flower", width, height, "flower")



#################  test ###################
example_cat_image <- readImage(file.path(img_dir,"cat_train_1.jpg"))
mean(example_cat_image@.Data[1,1,])
img_resized <- resize(example_cat_image, w = width, h = height)
grayimg <- channel(img_resized, "gray")
img_matrix <- grayimg@.Data
img_vector <- as.vector(t(img_matrix))
display(example_cat_image)
display(grayimg)

test_data_cat <- read_images(img_dir, width, height)
test_data_cat <- add_label(test_data_cat, "cat")

#################  test ###################

