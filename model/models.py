from torch import nn
import torch.nn.functional
import torch
from .ops.basic_ops import ConsensusModule, Identity
# from transforms import *
from torch.nn.init import normal, constant
from torch.nn import Parameter
import torchvision

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet18', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, modalities_fusion='cat', 
                 crop_num=1, partial_bn=True, context=False, embed=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.embed = embed

        self.name_base = base_model
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))
        self.embed = embed

        self._prepare_base_model(base_model)
        
        self.context = context

        if context:
            self._prepare_context_model()

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")

            if self.context:
                print("Converting the context model to a flow init model")
                self.context_model = self._construct_flow_model(self.context_model)
                print("Done. Flow model ready...")


        self.consensus = ConsensusModule(consensus_type)
        self.consensus_cont = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        std = 0.001

        if isinstance(self.base_model, torch.nn.modules.container.Sequential):
            feature_dim = 2048
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            if self.dropout == 0:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
                self.new_fc = None
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

        if self.embed:
            self.embed_fc = nn.Linear(feature_dim,300)
            normal(self.embed_fc.weight, 0, std)
            constant(self.embed_fc.bias, 0)

        if self.context:
            num_feats = 4096
        else:
            num_feats = 2048

        self.new_fc = nn.Linear(num_feats, num_class)
        normal(self.new_fc.weight, 0, std)
        constant(self.new_fc.bias, 0)

        self.new_fc_1 = nn.Linear(num_feats, 3)
        normal(self.new_fc_1.weight, 0, std)
        constant(self.new_fc_1.bias, 0)


        return num_feats


    def _prepare_context_model(self):
        self.context_model = getattr(torchvision.models, "resnet50")(True)
        modules = list(self.context_model.children())[:-1]  # delete the last fc layer.
        self.context_model = nn.Sequential(*modules)

    def _prepare_base_model(self, base_model):
        import torchvision, torchvision.models

        if 'resnet' in base_model or 'vgg' in base_model or 'resnext' in base_model or 'densenet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
       
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            count = 0
            if self.context:
                print("Freezing BatchNorm2D except the first one.")
                for m in self.context_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()

                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False



    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        params = [{'params': self.parameters()}]

        return params



    def forward(self, input, embeddings):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.context:
            inp = input.view((-1, sample_len) + input.size()[-2:])

            body_indices = list(range(0,inp.size(0),2))
            context_indices = list(range(1,inp.size(0),2))

            body = inp[body_indices]
            context = inp[context_indices]
        else:
            body = input.view((-1, sample_len) + input.size()[-2:])

        base_out = self.base_model(body).squeeze(-1).squeeze(-1)

        if self.context:
            context_out = self.context_model(context).squeeze(-1).squeeze(-1)
            base_out = torch.cat((base_out, context_out),dim=1)

        outputs = {}

        if self.embed:
            embed_segm = self.embed_fc(base_out)
            embed = embed_segm.view((-1, self.num_segments) + embed_segm.size()[1:])
            embed = self.consensus_embed(embed).squeeze(1)
            outputs['embed'] = embed


        base_out_cat = self.new_fc(base_out)
        base_out_cont = self.new_fc_1(base_out)

        base_out_cat = base_out_cat.view((-1, self.num_segments) + base_out_cat.size()[1:])
        base_out_cont = base_out_cont.view((-1, self.num_segments) + base_out_cont.size()[1:])

        output = self.consensus(base_out_cat)
        outputs['categorical'] = output.squeeze(1)

        output_cont = self.consensus_cont(base_out_cont)
        outputs['continuous'] = output_cont.squeeze(1)

        return outputs


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model


