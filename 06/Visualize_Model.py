import tensorflow.keras.backend as K
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


class ModelVisualizationClass(object):
    def __init__(self, model,verbose=True, save_images=False, out_path=os.path.join('.','Activations')):
        self.layer_names = None
        self.activation_model = None
        self.save_images = save_images
        self.out_path = out_path
        self.model = model
        all_layers = model.layers[:]
        self.all_layers = [layer for layer in all_layers if type(layer).__name__.lower().find('input') == -1] # and type(layer).__name__.lower().find('activation') != -1
        if verbose:
            self.print_all_layers()

    def print_all_layers(self):
        for layer in self.all_layers:
            print(layer.name)

    def define_desired_layers(self, desired_layer_names=None):
        all_layers = self.all_layers
        if desired_layer_names:
            all_layers = [layer for layer in all_layers if layer.name in desired_layer_names]
        self.layer_outputs = [layer.output for layer in all_layers]  # We already have the input.
        self.layer_names = [layer.name for layer in all_layers]  #
        self.activation_model = Model(inputs=self.model.input, outputs=self.layer_outputs)

    def predict_on_tensor(self, img_tensor):
        if self.activation_model is None:
            self.define_desired_layers()
        self.activations = self.activation_model.predict(img_tensor)
        if type(self.activations) != list:
            self.activations = [self.activations]

    def define_output(self,out_path):
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def plot_activations(self, ground_truth=None):
        assert self.activations is not None, 'Need to run predict_on_tensor first!'
        if not self.out_path and self.save_images:
            self.define_output(os.path.join('.','activation_outputs'))
        elif self.save_images:
            self.define_output(self.out_path)
        image_index = 0
        if self.layer_names is None:
            self.define_desired_layers()
        print(self.layer_names)
        if ground_truth is not None:
            ground_truth = np.argmax(np.squeeze(ground_truth),axis=-1)
        for layer_name, layer_activation in zip(self.layer_names, self.activations):
            layer_activation = np.squeeze(layer_activation)
            print(layer_name)
            print(self.layer_names.index(layer_name) / len(self.layer_names) * 100)
            if len(layer_activation.shape) == 4:
                middle_index = layer_activation.shape[0] // 2
                if ground_truth is not None:
                    index = np.where(ground_truth != 0)
                    if index:
                        indexes = np.unique(index[0])
                        middle_index = indexes[len(indexes)//2]
                layer_activation = layer_activation[middle_index,...]
            elif len(layer_activation.shape) == 5:
                layer_activation = layer_activation[0,...,0,:]
            display_grid = make_grid_from_map(layer_activation)
            scale = 0.01
            plt.figure(figsize=(display_grid.shape[1] * scale, scale * display_grid.shape[0]))
            plt.imshow(display_grid, aspect='auto', cmap='gray')
            plt.title(layer_name)
            plt.grid(False)
            if self.save_images:
                plt.savefig(os.path.join(self.out_path, '{}_{}.png'.format(image_index, layer_name.replace("/", '.'))))
                plt.close()
            image_index += 1

    def plot_activation(self, layer_name):
        if not self.out_path and self.save_images:
            self.define_output(os.path.join('.','activation_outputs'))
        elif self.save_images:
            self.define_output(self.out_path)
        image_index = 0
        print(self.layer_names)
        index = self.layer_names.index(layer_name)
        layer_activation = np.squeeze(self.activations[index])
        activation_shape = layer_activation.shape
        print(layer_name)
        if len(activation_shape) == 4:
            layer_activation = layer_activation[activation_shape[0]//2]
        elif len(activation_shape) == 5:
            layer_activation = layer_activation[0,...,0,:]
        display_grid = make_grid_from_map(layer_activation)
        scale = 0.05
        plt.figure(figsize=(display_grid.shape[1] * scale, scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')
        if self.save_images:
            plt.savefig(os.path.join(self.out_path, str(image_index) + '_' + layer_name + '.png'))
            plt.close()
        image_index += 1

    def make_grid_from_kernel(self, weights, image_index=0, layer_name=''):
        n_features = weights.shape[-1]
        split = 2
        while n_features / split % 2 == 0 and n_features / split >= split:
            split *= 2
        split /= 2
        images_per_row = int(n_features // split)
        n_cols = n_features // images_per_row
        for i in range(1, n_features + 1):
            plt.subplot(images_per_row, n_cols, i)
            weight = weights[..., i - 1]
            # weight = (weight-np.mean(weight))/np.std(weight)
            plt.imshow(weight, interpolation="nearest", cmap="gray")
        plt.show()
        plt.grid(False)
        if self.save_images:
            plt.savefig(os.path.join(self.out_path, str(image_index) + '_' + layer_name + '.png'))
            plt.close()
        return None

    def plot_kernels(self):
        if not self.out_path and self.save_images:
            self.define_output(os.path.join('.','kernel_outputs'))
        image_index = 0
        for layer_name in self.layer_names:
            print(layer_name)
            layer = [i for i in self.activation_model.layers if i.name == layer_name][0]
            kernels = layer.get_weights()[0]
            if len(kernels.shape) == 4:
                kernels = kernels[...,0,:]
            elif len(kernels.shape) == 5:
                kernels = kernels[0,...,0,:]
            self.make_grid_from_kernel(kernels, image_index=image_index,layer_name=layer_name)
            image_index += 1


def visualize_activations(model, img_tensor, out_path = os.path.join('.','activation_outputs')):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    all_layers = model.layers[1:]
    all_layers = [layer for layer in all_layers if layer.name.find('mask') == -1 and layer.name.lower().find('input') == -1 and layer.name.lower().find('batch_normalization') == -1]
    layer_outputs = [layer.output for layer in all_layers]  # We already have the input.
    layer_names = [layer.name for layer in all_layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    image_index = 0
    for layer_name, layer_activation in zip(layer_names, activations):
        print(layer_name)
        print(layer_names.index(layer_name)/len(layer_names) * 100)
        layer_activation = np.squeeze(layer_activation)
        display_grid = make_grid_from_map(layer_activation)
        scale = 0.05
        plt.figure(figsize=(display_grid.shape[1] * scale, scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')
        plt.savefig(os.path.join(out_path,str(image_index) + '_' + layer_name + '.png'))
        plt.close()
        image_index += 1


def make_grid_from_map(layer_activation):
    n_features = layer_activation.shape[-1]
    split = 2
    while n_features / split % 2 == 0 and n_features / split >= split:
        split *= 2
    split /= 2
    images_per_row = int(n_features // split)
    if len(layer_activation.shape) == 4:
        rows_size = layer_activation.shape[1]
        cols_size = layer_activation.shape[2]
    else:
        rows_size = layer_activation.shape[0]
        cols_size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((rows_size * images_per_row, n_cols * cols_size))
    for col in range(n_cols):
        for row in range(images_per_row):
            if len(layer_activation.shape) == 4:
                channel_image = layer_activation[layer_activation.shape[0] // 2, :, :, col * images_per_row + row]
            else:
                channel_image = layer_activation[:, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[row * rows_size: (row + 1) * rows_size,
            col * cols_size: (col + 1) * cols_size] = channel_image
    return display_grid


def decay_regularization(img, grads, decay = 0.9):
    return decay * img


def clip_weak_pixel_regularization(img, grads, percentile = 1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped


def gradient_ascent_iteration(loss_function, img):
    loss_value, grads_value = loss_function([img])
    gradient_ascent_step = img + grads_value * 0.9

    # Convert to row major format for using opencv routines
    grads_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
    img_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))

    # List of regularization functions to use
    regularizations = [decay_regularization, clip_weak_pixel_regularization]

    # The reguarlization weights
    weights = np.float32([3, 3, 1])
    weights /= np.sum(weights)

    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]
    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    img = np.sum(weighted_images, axis = 0)

    # Convert image back to 1 x 3 x height x width
    img = np.float32([np.transpose(img, (2, 0, 1))])

    return img


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(model, layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[..., filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    img = np.random.random((1, size, size, 3)) * 20 + 128.
    for i in range(30):
        img = gradient_ascent_iteration(iterate, img)
    return deprocess_image(img[0])


def visualize_filters(model):
    layer_name = 'block1_conv1'
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3) ,dtype='uint8')
    for i in range(8):
        print(i)
        for j in range(8):
            filter_img = generate_pattern(model, layer_name, i + (j * 8), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
    plt.figure(figsize=(20, 20))
    plt.imshow(results)


def main():
    pass


if __name__ == '__main__':
    main()
