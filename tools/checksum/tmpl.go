package main

var files = [...]string{
	"asset/Decoder.pt",
	"asset/DVAE.pt",
	"asset/GPT.pt",
	"asset/spk_stat.pt",
	"asset/tokenizer.pt",
	"asset/Vocos.pt",

	"config/decoder.yaml",
	"config/dvae.yaml",
	"config/gpt.yaml",
	"config/path.yaml",
	"config/vocos.yaml",
}

const jsontmpl = `{
	"sha256_asset_Decoder_pt"   : "%s",
	"sha256_asset_DVAE_pt"      : "%s",
	"sha256_asset_GPT_pt"       : "%s",
	"sha256_asset_spk_stat_pt"  : "%s",
	"sha256_asset_tokenizer_pt" : "%s",
	"sha256_asset_Vocos_pt"     : "%s",

	"sha256_config_decoder_yaml": "%s",
	"sha256_config_dvae_yaml"   : "%s",
	"sha256_config_gpt_yaml"    : "%s",
	"sha256_config_path_yaml"   : "%s",
	"sha256_config_vocos_yaml"  : "%s"
}
`
