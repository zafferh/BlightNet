IM_SIZE = 120
BATCH_SIZE_LTPA = 8
EPOCHS_LTPA = 50
LR_LTPA = 0.0001
ATTN_MODE_LTPA = 'after'
LOG_DIR = 'logs'
MODEL_DIR_LTPA = 'logs/net_49_0.75.pth'
BATCH_SIZE_BASE = 32
EPOCHS_BASE = 50
MODEL_DIR_BASE = 'logs\weight-improvement-40-0.75.hdf5'
LR_BASE = 1e-4
MOMENTUM_BASE = 0.9
WINDOW_SIZES = [(224, 224), (168,168), (112,112)]
STRIDES = [32, 44, 48]
BATCH_SIZE_CLTPA = 4
EPOCHS_CLTPA = 8
LR_CLTPA = 0.0001
ATTN_MODE_CLTPA = 'after'
MODEL_DIR_CLTPA = 'logs/net_49_0.75.pth'
CLTPA_MODEL_LIST = [
	'logs/net_49_0.71_0_0_224.pth',
	'logs/net_49_0.66_0_32_224.pth',
	'logs/net_49_0.69_32_0_224.pth',
	'logs/net_49_0.73_32_32_224.pth',
	'logs/net_49_0.68_0_0_168.pth',
	'logs/net_49_0.68_0_44_168.pth',
	'logs/net_49_0.69_0_88_168.pth',
	'logs/net_49_0.71_44_0_168.pth',
	'logs/net_49_0.66_44_44_168.pth',
	'logs/net_49_0.65_44_88_168.pth',
	'logs/net_49_0.76_88_0_168.pth',
	'logs/net_49_0.69_88_44_168.pth',
	'logs/net_49_0.67_88_88_168.pth',
	'logs/net_49_0.69_0_0_112.pth',
	'logs/net_49_0.64_0_48_112.pth',
	'logs/net_49_0.65_0_96_112.pth',
	'logs/net_49_0.69_0_144_112.pth',
	'logs/net_49_0.67_48_0_112.pth',
	'logs/net_49_0.69_48_48_112.pth',
	'logs/net_49_0.65_48_96_112.pth',
	'logs/net_49_0.66_48_144_112.pth',
	'logs/net_49_0.69_96_0_112.pth',
	'logs/net_49_0.69_96_48_112.pth',
	'logs/net_49_0.72_96_96_112.pth',
	'logs/net_49_0.66_96_144_112.pth',
	'logs/net_49_0.68_144_0_112.pth',
	'logs/net_49_0.70_144_48_112.pth',
	'logs/net_49_0.73_144_96_112.pth',
	'logs/net_49_0.67_144_144_112.pth'
]
