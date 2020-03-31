from model.sadnet import SADNET


def make_model(input_channel, output_channel, args):
    if args.NetName == 'SADNET':
        return SADNET(input_channel, output_channel, args.n_channel, args.offset_channel)
