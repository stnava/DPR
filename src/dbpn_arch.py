
from keras.models import Model
from keras.layers import (Input, Add, Subtract,
                          PReLU, Concatenate,
                          UpSampling2D, UpSampling3D,
                          Conv2D, Conv2DTranspose,
                          Conv3D, Conv3DTranspose)


# define the DBPN network - this uses a model definition that is general to<br>
# both 2D and 3D. recommended parameters for different upsampling rates can<br>
# be found in the papers by Haris et al.  We make one significant change to<br>
# the original architecture by allowing standard interpolation for upsampling<br>
# instead of convolutional upsampling.  this is controlled by the interpolation<br>
# option.

# In[74]:


def dbpn(input_image_size,
                                                 number_of_outputs=1,
                                                 number_of_base_filters=64,
                                                 number_of_feature_filters=256,
                                                 number_of_back_projection_stages=7,
                                                 convolution_kernel_size=(12, 12),
                                                 strides=(8, 8),
                                                 last_convolution=(3, 3),
                                                 number_of_loss_functions=1,
                                                 interpolation = 'nearest'
                                                ):
    idim = len( input_image_size ) - 1
    if idim == 2:
        myconv = Conv2D
        myconv_transpose = Conv2DTranspose
        myupsampling = UpSampling2D
        shax = ( 1, 2 )
        firstConv = (3,3)
        firstStrides=(1,1)
        smashConv=(1,1)
    if idim == 3:
        myconv = Conv3D
        myconv_transpose = Conv3DTranspose
        myupsampling = UpSampling3D
        shax = ( 1, 2, 3 )
        firstConv = (3,3,3)
        firstStrides=(1,1,1)
        smashConv=(1,1,1)
    def up_block_2d(L, number_of_filters=64, kernel_size=(12, 12), strides=(8, 8),
                    include_dense_convolution_layer=True):
        if include_dense_convolution_layer == True:
            L = myconv(filters = number_of_filters,
                       use_bias=True,
                       kernel_size=smashConv,
                       strides=firstStrides,
                       padding='same')(L)
            L = PReLU(alpha_initializer='zero',
                      shared_axes=shax)(L)
        # Scale up
        if idim == 2:
            H0 = myupsampling( size = strides, interpolation=interpolation )(L)
        if idim == 3:
            H0 = myupsampling( size = strides )(L)
        H0 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H0)
        H0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H0)
        # Scale down
        L0 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(H0)
        L0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L0)
        # Residual
        E = Subtract()([L0, L])
        # Scale residual up
        if idim == 2:
            H1 = myupsampling( size = strides, interpolation=interpolation  )(E)
        if idim == 3:
            H1 = myupsampling( size = strides )(E)
        H1 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H1)
        H1 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H1)
        # Output feature map
        up_block = Add()([H0, H1])
        return up_block
    def down_block_2d(H, number_of_filters=64, kernel_size=(12, 12), strides=(8, 8),
                    include_dense_convolution_layer=True):
        if include_dense_convolution_layer == True:
            H = myconv(filters = number_of_filters,
                       use_bias=True,
                       kernel_size=smashConv,
                       strides=firstStrides,
                       padding='same')(H)
            H = PReLU(alpha_initializer='zero',
                      shared_axes=shax)(H)
        # Scale down
        L0 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(H)
        L0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L0)
        # Scale up
        if idim == 2:
            H0 = myupsampling( size = strides, interpolation=interpolation )(L0)
        if idim == 3:
            H0 = myupsampling( size = strides )(L0)
        H0 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H0)
        H0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H0)
        # Residual
        E = Subtract()([H0, H])
        # Scale residual down
        L1 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(E)
        L1 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L1)
        # Output feature map
        down_block = Add()([L0, L1])
        return down_block
    inputs = Input(shape=input_image_size)
    # Initial feature extraction
    model = myconv(filters=number_of_feature_filters,
                   kernel_size=firstConv,
                   strides=firstStrides,
                   padding='same',
                   kernel_initializer='glorot_uniform')(inputs)
    model = PReLU(alpha_initializer='zero',
                  shared_axes=shax)(model)
    # Feature smashing
    model = myconv(filters=number_of_base_filters,
                   kernel_size=smashConv,
                   strides=firstStrides,
                   padding='same',
                   kernel_initializer='glorot_uniform')(model)
    model = PReLU(alpha_initializer='zero',
                  shared_axes=shax)(model)
    # Back projection
    up_projection_blocks = []
    down_projection_blocks = []
    model = up_block_2d(model, number_of_filters=number_of_base_filters,
      kernel_size=convolution_kernel_size, strides=strides)
    up_projection_blocks.append(model)
    for i in range(number_of_back_projection_stages):
        if i == 0:
            model = down_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides)
            down_projection_blocks.append(model)
            model = up_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides)
            up_projection_blocks.append(model)
            model = Concatenate()(up_projection_blocks)
        else:
            model = down_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides,
              include_dense_convolution_layer=True)
            down_projection_blocks.append(model)
            model = Concatenate()(down_projection_blocks)
            model = up_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides,
              include_dense_convolution_layer=True)
            up_projection_blocks.append(model)
            model = Concatenate()(up_projection_blocks)
    outputs = myconv(filters=number_of_outputs,
                     kernel_size=last_convolution,
                     strides=firstStrides,
                     padding = 'same',
                     kernel_initializer = "glorot_uniform")(model)
    if number_of_loss_functions == 1:
        deep_back_projection_network_model = Model(inputs=inputs, outputs=outputs)
    else:
        outputList=[]
        for k in range(number_of_loss_functions):
            outputList.append(outputs)
        deep_back_projection_network_model = Model(inputs=inputs, outputs=outputList)
    return deep_back_projection_network_model

