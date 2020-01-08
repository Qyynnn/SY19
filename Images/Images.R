library(EBImage)
library(pbapply)
library(dplyr)
library(tensorflow)
library(reticulate)
reticulate::py_config()
import("scipy")

setwd("./Données du TP 10-20191217/images_train")

width <- 64
height <- 64

#############################################################
############ 1. Image loading and Pre-processing ############
#############################################################

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

# load images

cats_data <- read_images("./cat", width, height, "cat") # 0
cars_data <- read_images("./car", width, height, "car") # 1
flowers_data <- read_images("./flower", width, height, "flower") # 2

# save data
saveRDS(cats_data, "cat.rds")
saveRDS(cars_data, "dog.rds")
saveRDS(flowers_data, "flower.rds")

###########################################
############ 2. Model Training ############
###########################################

library(caret)
set.seed(1749)

## combine three dataset together
data_complete <- rbind(cats_data, cars_data, flowers_data)
data_complete$label <- as.factor(data_complete$label)
saveRDS(data_complete, "data_complet.rds")


## create training set
training_index <- createDataPartition(data_complete$label, p = 0.9, times = 1)
training_index <- unlist(training_index)

train_set <- data_complete[training_index,]
dim(train_set)

test_set <- data_complete[-training_index,]
dim(test_set)


################################################################
############ 2.1 convolutional neural network (CNN) ############
################################################################

library(keras)

# Reshape data
train_data <- data.matrix(train_set)
train_x <- t(train_data[,-1])
train_y <- train_data[,1]-1
train_y <- to_categorical(train_y)

train_array <- train_x
dim(train_array) <- c(width, height, 1, ncol(train_x))
train_array <- aperm(train_array, c(4,1,2,3))

test_data <- data.matrix(test_set)
test_x <- t(test_data[,-1])
test_y <- test_data[,1]-1
test_y <- to_categorical(test_y)

test_array <- test_x
dim(test_array) <- c(width, width, 1, ncol(test_x))
test_array <- aperm(test_array, c(4,1,2,3))

# check the image 
example_cat <- test_array[32,,,]
image(t(apply(example_cat, 2, rev)), col = gray.colors(12),
      axes = F)

# Define the model CNN
model_cnn <- keras_model_sequential()
model_cnn %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(width, height, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 3, activation = "softmax")

summary(model_cnn)

model_cnn %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Data augmentation and training
data_augmentation <- FALSE

if(!data_augmentation){
  fit_cnn <- model_cnn %>% fit(
    x = train_array, 
    y = train_y,
    epochs = 50, 
    batch_size = 128,
    verbose = 2, 
    validation_split = 0.1
  )
  plot(fit_cnn)
}else{
  # generating images
  gen_images <- image_data_generator(rotation_range = 20,
                                     width_shift_range = 0.20,
                                     height_shift_range = 0.20,
                                     horizontal_flip = TRUE,
                                     validation_split = 0.1)
  
  # Fit image data generator internal statistics to training data
  gen_images %>% fit_image_data_generator(train_array)
  # Generates batches of augmented/normalized data from image data and labels
  # to visually see the generated images by the Model
  fit_cnn <- model_cnn %>% 
    fit_generator(
      flow_images_from_data(train_array,
                            train_y,
                            generator = gen_images,
                            batch_size = 128),
      steps_per_epoch = as.integer(ncol(train_x)/128),
      epochs = 50
      )
  plot(fit_cnn)
}


# Test on test datasets
predictions_cnn <-  predict_classes(model_cnn, test_array)
probabilities_cnn <- predict_proba(model_cnn, test_array)

# Visualisation
set.seed(1749)
random <- sample(1:nrow(test_data), 32)
preds_cnn <- predictions_cnn[random]
probs_cnn <- as.vector(round(probabilities_cnn[random,], 2))

par(mfrow = c(4, 8), mar = rep(0, 4))
for(i in 1:length(random)){
  image(t(apply(test_array[random[i],,,], 2, rev)),
        col = gray.colors(12), axes = F)
  legend("topright", legend = ifelse(preds_cnn[i] == 0, "Cat", 
                                     ifelse(preds_cnn[i] == 1, "Car", "Flower")),
         text.col = ifelse(preds_cnn[i] == 0, 2,
                           ifelse(preds_cnn[i] == 1, 4, 3)), 
         bty = "n", text.font = 2)
  legend("topleft", legend = probs_cnn[i], bty = "n", col = "white")
}

save(model_cnn, file = "CNNmodel.RData")


################################################################
############### 2.2 MLP (Multilayer Perceptron) ################
################################################################

# Reshape the train_array, train_y and test_arrary and test_y
train_array <- array_reshape(train_x, c(ncol(train_x), width * height))
# train_y and test_y the same as CNN
test_array <- array_reshape(test_x, c(ncol(test_x), width * height))

# Build model
model_mlp = keras_model_sequential() %>% 
  layer_dense(units = 512,               # number of perceptron
              activation = "relu",       # activation function
              input_shape = c(4096)       # dimensions of input tensor
              ) %>% 
  layer_dense(units = 128,
              activation = "relu"
              ) %>% 
  layer_dense(units = 3,                 # one output neuron per class # 最后一层类别数
              activation = "softmax"     # activate the largest one
              )
summary(model_mlp)

model_mlp %>% compile(                 # specify
  optimizer = "adam",               # optimizer
  loss = "categorical_crossentropy",   # loss function
  metrics = "accuracy"           # accuracy metrice  
)


# Train model
fit_mlp <- model_mlp %>% fit(
  train_array,         # train data
  train_y,             # label of train data
  epochs=30,            # no. epochs # nombre de training
  batch_size=128,       # no. input per mini-batch # batch_size大跑比较快(GPU要大)
  verbose=2,             # 每個epoch打一個進度
  validation_split = 0.1
)
plot(fit_mlp)

# predictions
predictions_mlp <-  predict_classes(model_mlp, test_array)
probabilities_mlp <- predict_proba(model_mlp, test_array)

preds_mlp <- predictions_mlp[random]
probs_mlp <- as.vector(round(probabilities_mlp[random,], 2))

par(mfrow = c(4, 8), mar = rep(0, 4))
test_array <- test_x
dim(test_array) <- c(width, width, 1, ncol(test_x))
test_array <- aperm(test_array, c(4,1,2,3))
for(i in 1:length(random)){
  image(t(apply(test_array[random[i],,,], 2, rev)),
        col = gray.colors(12), axes = F)
  legend("topright", legend = ifelse(preds_mlp[i] == 0, "Cat", 
                                     ifelse(preds_mlp[i] == 1, "Car", "Flower")),
         text.col = ifelse(preds_mlp[i] == 0, 2,
                           ifelse(preds_mlp[i] == 1, 4, 3)), 
         bty = "n", text.font = 2)
  legend("topleft", legend = probs_mlp[i], bty = "n", col = "white")
}


