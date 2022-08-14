# -*- coding: utf8 -*-
"""
base NN blocks
"""
import tensorflow as tf
from . import custom_layers


class BaseArchitecture(object):

    @staticmethod
    def _get_allocate_layer(name, type_, model, **kwargs):
        """ Allocate new layer or find existing """

        # check layer is allocated in model
        if model is not None:
            return model.get_layer(name)

        # Allocate new layer
        if type_ == 'batch':
            return tf.keras.layers.BatchNormalization(name=name, **kwargs)
        elif type_ == 'dense':
            node = kwargs.pop('node')
            # print("DEBUG 24 ", node, name, kwargs)
            return tf.keras.layers.Dense(node, name=name, **kwargs)
        elif type_ == 'drop':
            rate = kwargs.pop('drop')
            drop_class = kwargs.pop('drop_class')
            drop_layer = getattr(tf.keras.layers, drop_class)
            return drop_layer(rate, name=name, **kwargs)
        elif type_ == 'relu':
            return tf.keras.layers.ReLU(name=name)
        elif type_ == 'leaky':
            alpha = kwargs.pop('alpha')
            return tf.keras.layers.LeakyReLU(alpha, name=name)
        elif type_ == 'add':
            return tf.keras.layers.Add(name=name)
        elif type_ == 'identity':
            return custom_layers.Identity(name=name)

    @staticmethod
    def _build_layers(nodes, dropouts, dense_activation, suff='', model=None, disable_bn=False, dense_props=None,
                      drop_props=None, act_props=None, batch_props=None, rnd=None):

        """ Build NN branch from provided config

        Parameters
        ----------
        rnd
        disable_bn
        """

        if dense_activation not in ['relu', 'leaky', 'linear', None]:
            raise ValueError("Dense layer activation `%s` is not known" % dense_activation)

        # make nodes and droputs same length
        if len(dropouts) < len(nodes):
            dropouts += [None] * (len(nodes) - len(dropouts))

        bn_l = []
        dense_l = []
        drop_l = []
        activations_l = []

        if dense_props is None:
            dense_props = dict()

        if drop_props is None:
            drop_props = dict()

        if act_props is None:
            act_props = dict()

        if batch_props is None:
            batch_props = dict()

        for index_layer, node in enumerate(nodes):

            # Dropouts
            name_ = 'BN_%s%d' % (suff, index_layer)
            if disable_bn:
                bn_l.append(None)
            else:
                bn_l.append(BaseArchitecture._get_allocate_layer(name_, 'batch', model, **batch_props))

            # Dense
            name_ = 'Den_%s%d' % (suff, index_layer)

            # init seed if random generator provided
            dense_props_mod = BaseArchitecture._set_seed_dense_initializer(dense_props, rnd)
            dense_props_mod['activation'] = 'linear'  # place activation as external layer
            dense_l.append(
                BaseArchitecture._get_allocate_layer(name_, 'dense', model, node=node, **dense_props_mod)
            )

            # Dropouts
            name_ = 'Drop_%s%d' % (suff, index_layer)
            dropout = dropouts[index_layer]
            if dropout is None:
                drop_l.append(None)
            else:
                drop_l.append(BaseArchitecture._get_allocate_layer(name_, 'drop', model, drop=dropout, **drop_props))

            # Activations
            name_ = 'Act_%s%d' % (suff, index_layer)
            if dense_activation == 'linear':
                act_l = None
            else:
                act_l = BaseArchitecture._get_allocate_layer(name_, dense_activation, model, **act_props)
            activations_l.append(act_l)

        return bn_l, dense_l, drop_l, activations_l

    @staticmethod
    def _set_seed_dense_initializer(dense_props, rnd):
        if rnd is not None:
            dense_props = dict(dense_props)
            initializer = getattr(tf.keras.initializers, dense_props['kernel_initializer'])
            initializer = initializer(seed=rnd.randint(2 ** 30))
            dense_props['kernel_initializer'] = initializer
        return dense_props

    @staticmethod
    def _dense_act(input_, dense, act, drop):
        """ Dense -> Activation block -> Drop """
        input_ = dense(input_)
        input_ = act(input_) if act is not None else input_
        input_ = drop(input_) if drop is not None else input_
        return input_

    @staticmethod
    def _dense_bn_act(input_, dense, bn, act, drop):
        """ Dense -> Drop -> BatchNorm -> Activation block """
        input_ = dense(input_)
        input_ = drop(input_) if drop is not None else input_
        input_ = bn(input_) if bn is not None else input_
        input_ = act(input_) if act is not None else input_
        return input_

    @staticmethod
    def _bn_act_dense(input_, dense, bn, act, drop):
        """ Drop -> BatchNorm -> Activation block -> Dense block"""
        input_ = drop(input_) if drop is not None else input_
        input_ = bn(input_) if bn is not None else input_
        input_ = act(input_) if act is not None else input_
        input_ = dense(input_)
        return input_

    @staticmethod
    def _assemble_nn_unit(input_, dense_layers, bn_layers,
                          drop_layers, act_layers, bn_first=False, disable_bn=False):
        """ Construct base neural network units from lists of dense, activation, dropouts and BatchNorm layers """
        assert len(dense_layers) == len(bn_layers) == len(drop_layers) == len(act_layers)
        for i, layer in enumerate(dense_layers):
            with tf.name_scope('Unit%d' % i):
                if disable_bn:
                    input_ = BaseArchitecture._dense_act(input_, dense_layers[i],
                                                         act_layers[i], drop_layers[i])
                elif bn_first:
                    input_ = BaseArchitecture._bn_act_dense(input_, dense_layers[i], bn_layers[i],
                                                            act_layers[i], drop_layers[i])
                else:
                    input_ = BaseArchitecture._dense_bn_act(input_, dense_layers[i], bn_layers[i],
                                                            act_layers[i], drop_layers[i])
        return input_

    @staticmethod
    def _build_branch(input_, **config):
        """ Build branch of blocks for Dense AE or ResNet AE """

        nodes = config.get('nodes')
        dropouts = config.get('dropouts')
        # dense_activation = config.get('dense_activation')
        suff = config.get('suff')
        bn_first = config.get('bn_first')
        disable_bn = config.get('disable_bn')
        model = config.get('model', None)
        dense_props = config.get('dense_props')
        drop_props = config.get('drop_props')
        act_props = config.get('act_props')
        batch_props = config.get('batch_props')
        rnd = config.get('rnd', None)
        activation = dense_props.pop('activation')

        if len(nodes) == 0:
            return input_

        assert not (None in nodes), "All nodes should be defined for dense AE"
        batch_l, dense_l, drops_l, activation_l = \
            BaseArchitecture._build_layers(nodes, dropouts, activation, suff=suff, model=model,
                                           disable_bn=disable_bn, dense_props=dense_props, drop_props=drop_props,
                                           act_props=act_props, batch_props=batch_props, rnd=rnd)

        encoded = BaseArchitecture._assemble_nn_unit(input_, dense_l, batch_l,
                                                     drops_l, activation_l, bn_first, disable_bn)

        return encoded


class ResNetAE(BaseArchitecture):
    @staticmethod
    def _resnet_block(input_, nodes, dropouts, dense_activation, bn_first=True, drop_end=None, ib=0, pref='',
                      model=None, disable_bn=False,
                      dense_props=None, drop_props=None, act_props=None, batch_props=None, rnd=None):
        """ One ResNet block, double """

        input_shape = int(input_.shape[1])
        extended_nodes = nodes + [input_shape]
        batch_l, dense_l, drops_l, activation_l = \
            ResNetAE._build_layers(
                extended_nodes, dropouts, dense_activation, '%sRes%d' % (pref, ib), model, disable_bn,
                dense_props=dense_props, drop_props=drop_props, act_props=act_props, batch_props=batch_props, rnd=rnd
            )

        resudial_ = ResNetAE._assemble_nn_unit(input_, dense_l, batch_l, drops_l, activation_l, bn_first, disable_bn)

        add_l = BaseArchitecture._get_allocate_layer("%sAdd_Res%d" % (pref, ib), 'add', model)
        resudial_ = add_l([input_, resudial_])

        if drop_end is not None:
            drop_l = BaseArchitecture._get_allocate_layer(
                '%sDrop_Res%d' % (pref, ib), 'drop', model, drop=drop_end, **drop_props
            )
            resudial_ = drop_l(resudial_)
        return resudial_

    @staticmethod
    def _resnet_chain(input_, nodes, resnet_nodes, dropouts, dense_activation, bn_first=True, suff='', model=None,
                      disable_bn=False, dense_props=None, drop_props=None, act_props=None, batch_props=None, rnd=None):

        assert len(nodes) == len(resnet_nodes), "Wrong ResNet Config"
        assert len(nodes) == len(dropouts), "Wrong Dropout Config"

        input_shape = int(input_.shape[1])
        for i, node in enumerate(nodes):
            if (node is None) or (input_shape == node):
                #     print ("You are adding Dense layer without reducing dimension. Use `None` in \
                #         nodes config to repeat dimension of the previous layer. Example nodes=[5, None, None]")
                pass
            else:
                # init seed if random generator provided
                dense_props_mod = BaseArchitecture._set_seed_dense_initializer(dense_props, rnd)
                dense_props_mod['activation'] = 'linear'
                dense_l = BaseArchitecture._get_allocate_layer(
                    'Den_RedRes%s%d' % (suff, i), 'dense', model, node=node, **dense_props_mod
                )
                input_ = dense_l(input_)

            resnet_node = resnet_nodes[i]

            with tf.name_scope("ResBlock%s%d" % (suff, i)):
                input_ = ResNetAE._resnet_block(
                    input_, resnet_node, [None] * len(resnet_node), dense_activation,
                    bn_first=bn_first, drop_end=dropouts[i], ib=i, pref=suff, model=model,
                    disable_bn=disable_bn, dense_props=dense_props, drop_props=drop_props,
                    act_props=act_props, batch_props=batch_props, rnd=rnd
                )

            input_shape = int(input_.shape[1])
        return input_

    @staticmethod
    def _build_branch(input_, **config):
        """ Build branch of blocks for Dense AE or ResNet AE """

        nodes = config.get('nodes')
        # use_bias_dense = config.get('use_bias_dense')
        dropouts = config.get('dropouts')
        # leaky_alpha = config.get('leaky_alpha')
        # dense_activation = config.get('dense_activation')
        suff = config.get('suff')
        bn_first = config.get('bn_first')
        disable_bn = config.get('disable_bn')
        Resnet_config = config.get('Resnet_config', None)
        model = config.get('model', None)
        dense_props = config.get('dense_props')
        drop_props = config.get('drop_props')
        act_props = config.get('act_props')
        batch_props = config.get('batch_props')
        rnd = config.get('rnd', None)
        activation = dense_props.pop('activation')
        if len(nodes) == 0:
            return input_

        if Resnet_config is None:
            raise ValueError('Resnet_config should be provided if ResNet architecture is used')

        # Swap first None with the non-None
        input_shape = int(input_.shape[1])
        if nodes[0] is None:
            for i, elem in enumerate(nodes[1:]):
                if elem is not None:
                    nodes[0], nodes[i + 1] = nodes[i + 1], nodes[0]
                    if nodes[0] == input_shape:
                        nodes[0] = None
                    break
        encoded = ResNetAE._resnet_chain(
            input_, nodes, Resnet_config, dropouts, activation, bn_first, suff=suff,  # TODO
            model=model, disable_bn=disable_bn, dense_props=dense_props, drop_props=drop_props,
            act_props=act_props, batch_props=batch_props, rnd=rnd
        )
        return encoded
