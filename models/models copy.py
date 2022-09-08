import tensorflow as tf

def res_block(x, size, strides=1, reg = None, name = ''):
    idt = tf.keras.layers.Conv2D(size, (1,1), padding='same', strides=strides, kernel_regularizer=reg, bias_regularizer=reg, name = f'{name}_conv_idt')(x)
    x = bn_relu(x, name = f'{name}_bnrelu_1')
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(size, (3,3), padding='same', strides = strides, kernel_regularizer=reg, bias_regularizer=reg, name = f'{name}_conv_1')(x)
    x = bn_relu(x, name = f'{name}_bnrelu_2')
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(size, (3,3), padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = f'{name}_conv_2')(x)
    return tf.keras.layers.Add(name = f'{name}_add')([x, idt])

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, size, strides, name = ''):
        super().__init__()

        self.idt_conv =  tf.keras.layers.Conv2D(size, (1,1), padding='same', strides=strides, name = f'{name}_conv_idt')
        self.bn_relu_0 = BN_Relu(name = f'{name}_bn_relu_0')
        self.conv_0 = tf.keras.layers.Conv2D(size, (3,3), padding='same', strides = strides, name = f'{name}_conv_0')

        self.bn_relu_1 = BN_Relu(name = f'{name}_bn_relu_1')
        self.conv_1 = tf.keras.layers.Conv2D(size, (3,3), padding='same', name = f'{name}_conv_1')

        self.add = tf.keras.layers.Add(name = f'{name}_add')

    def call(self, inputs, training = None):
        idt = self.idt_conv(inputs)
        x = self.bn_relu_0(inputs, training)
        x = self.conv_0(x)

        x = self.bn_relu_1(x, training)
        x = self.conv_1(x)

        return self.add([idt, x])

class BN_Relu(tf.keras.layers.Layer):
    def __init__(self, name = ''):
        super().__init__()
        self.bn = tf.keras.layers.BatchNormalization(name = f'{name}_bn')

    def call(self, x, training = None):
        x = self.bn(x, training)
        return tf.keras.activations.relu(x)

def bn_relu(x, name = ''):
    x = tf.keras.layers.BatchNormalization(name = f'{name}_bn')(x)
    return tf.keras.layers.Activation('relu', name=f'{name}_relu')(x)

def resunet_encoder(input_layer, model_size, reg_weight, name):
    x = tf.keras.layers.Conv2D(model_size[0], (3,3), padding='same', name = f'{name}_e0_conv_1')(input_layer)
    x = bn_relu(x, name = f'{name}_e0_bnrelu')
    x = tf.keras.layers.Conv2D(model_size[0], (3,3), padding='same', name = f'{name}_e0_conv_2')(x)
    idt = tf.keras.layers.Conv2D(model_size[0], (1,1), padding='same', name = f'{name}_e0_conv_idt')(input_layer)
    e1 = tf.keras.layers.Add(name = f'{name}_e0_add')([x, idt])
    
    e2 = res_block(e1, model_size[1], 2, name = f'{name}_e1')
    
    e3 = res_block(e2, model_size[2], 2, name = f'{name}_e2')
    
    bt = res_block(e3, model_size[3], 2, name = f'{name}_e3')

    return bt, e3, e2, e1

def resunet_decoder(encoder_outs, model_size, reg_weight, name):
    bt, e3, e2, e1 = encoder_outs

    d3 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_3')(bt)
    if e3 is not None: d3 = tf.keras.layers.Concatenate(name = f'{name}_concat_3', axis=-1)([d3, e3])
    
    d2 = res_block(d3, model_size[2], 1, name = f'{name}_d3')
    
    d2 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_2')(d2)
    if e2 is not None: d2 = tf.keras.layers.Concatenate(name = f'{name}_concat_2', axis=-1)([d2, e2])
    
    d1 = res_block(d2, model_size[1], 1, name = f'{name}_d2')
    
    d1 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_1')(d1)
    if e1 is not None: d1 = tf.keras.layers.Concatenate(name = f'{name}_concat_1', axis=-1)([d1, e1])
    
    d0 = res_block(d1, model_size[0], 1, name = f'{name}_d1')

    return d0

def resunet_classifier(decoder_outs, model_size, n_output, reg_weight, name):
    fusion_output = res_block(decoder_outs, model_size[0], 1, name = f'{name}_resblock')
    output = tf.keras.layers.Conv2D(n_output, (1,1), padding='same', activation='softmax', name = f'{name}_lastconv_softmax')(fusion_output)

    return output


def late_fusion_resunet(shape_opt, shape_sar, model_size, n_output, reg_weight, name):

    #optical stream
    opt_input_0= tf.keras.Input(shape_opt, name=f'{name}_opt_input_0') 
    opt_input_1= tf.keras.Input(shape_opt, name=f'{name}_opt_input_1') 
    opt_input = tf.concat([opt_input_0,  opt_input_1], axis=-1)
    opt_encoder_outs = resunet_encoder(opt_input, model_size, reg_weight, name=f'{name}_opt_encoder')
    opt_decoder_outs = resunet_decoder(opt_encoder_outs, model_size, reg_weight, name = f'{name}_opt_decoder')

    #sar encoder
    sar_input_0= tf.keras.Input(shape_sar, name=f'{name}_sar_input_0')
    sar_input_1= tf.keras.Input(shape_sar, name=f'{name}_sar_input_1')
    sar_input = tf.concat([sar_input_0,  sar_input_1], axis=-1)
    sar_encoder_outs = resunet_encoder(sar_input, model_size, reg_weight, name=f'{name}_sar_encoder')
    sar_decoder_outs = resunet_decoder(sar_encoder_outs, model_size, reg_weight, name = f'{name}_sar_decoder')

    #fusion
    fus_decoder_outs = tf.keras.layers.Concatenate(name = f'{name}_concat_fus', axis = -1)([opt_decoder_outs, sar_decoder_outs])
    fus_classifier_out = resunet_classifier(fus_decoder_outs, model_size, n_output, reg_weight, name = f'{name}_fus_classifier')

    return tf.keras.models.Model(inputs=[opt_input_0, opt_input_1, sar_input_0, sar_input_1], outputs = fus_classifier_out)


def sm_resunet(shape, model_size, n_output, reg_weight, name):

    #optical stream
    input_0= tf.keras.Input(shape, name=f'{name}_opt_input_0') 
    input_1= tf.keras.Input(shape, name=f'{name}_opt_input_1') 
    input = tf.concat([input_0,  input_1], axis=-1)
    encoder_outs = resunet_encoder(input, model_size, reg_weight, name=f'{name}_opt_encoder')
    decoder_outs = resunet_decoder(encoder_outs, model_size, reg_weight, name = f'{name}_opt_decoder')

    fus_classifier_out = resunet_classifier(decoder_outs, model_size, n_output, reg_weight, name = f'{name}_fus_classifier')

    return tf.keras.models.Model(inputs=[input_0, input_1], outputs = fus_classifier_out)


def late_fusion_resunet_previous(shape_opt, shape_sar, shape_previous, model_size, n_output, reg_weight, name):

    previous_input= tf.keras.Input(shape_previous, name=f'{name}_previous_input') 
    #optical stream
    opt_input_0= tf.keras.Input(shape_opt, name=f'{name}_opt_input_0') 
    opt_input_1= tf.keras.Input(shape_opt, name=f'{name}_opt_input_1') 
    opt_input = tf.concat([opt_input_0,  opt_input_1, previous_input], axis=-1)
    opt_encoder_outs = resunet_encoder(opt_input, model_size, reg_weight, name=f'{name}_opt_encoder')
    opt_decoder_outs = resunet_decoder(opt_encoder_outs, model_size, reg_weight, name = f'{name}_opt_decoder')

    #sar encoder
    sar_input_0= tf.keras.Input(shape_sar, name=f'{name}_sar_input_0')
    sar_input_1= tf.keras.Input(shape_sar, name=f'{name}_sar_input_1')
    sar_input = tf.concat([sar_input_0,  sar_input_1, previous_input], axis=-1)
    sar_encoder_outs = resunet_encoder(sar_input, model_size, reg_weight, name=f'{name}_sar_encoder')
    sar_decoder_outs = resunet_decoder(sar_encoder_outs, model_size, reg_weight, name = f'{name}_sar_decoder')

    #fusion
    fus_decoder_outs = tf.keras.layers.Concatenate(name = f'{name}_concat_fus', axis = -1)([opt_decoder_outs, sar_decoder_outs])
    fus_classifier_out = resunet_classifier(fus_decoder_outs, model_size, n_output, reg_weight, name = f'{name}_fus_classifier')

    return tf.keras.models.Model(inputs=[opt_input_0, opt_input_1, sar_input_0, sar_input_1, previous_input], outputs = fus_classifier_out)

def sm_resunet_pm(shape, shape_previous, model_size, n_output, reg_weight, name):

    #optical stream
    input_0= tf.keras.Input(shape, name=f'{name}_opt_input_0') 
    input_1= tf.keras.Input(shape, name=f'{name}_opt_input_1') 
    previous_input= tf.keras.Input(shape_previous, name=f'{name}_previous_input') 
    #input = tf.concat([input_0,  input_1], axis=-1)
    input = tf.concat([input_0,  input_1, previous_input], axis=-1)
    encoder_outs = resunet_encoder(input, model_size, reg_weight, name=f'{name}_opt_encoder')
    decoder_outs = resunet_decoder(encoder_outs, model_size, reg_weight, name = f'{name}_opt_decoder')

    fus_classifier_out = resunet_classifier(decoder_outs, model_size, n_output, reg_weight, name = f'{name}_fus_classifier')

    return tf.keras.models.Model(inputs=[input_0, input_1, previous_input], outputs = fus_classifier_out)