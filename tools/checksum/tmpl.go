package main

var files = [...]string{
	"asset/Decoder.pt",
	"asset/DVAE_full.pt",
	"asset/GPT.pt",
	"asset/spk_stat.pt",
	"asset/tokenizer.pt",
	"asset/Vocos.pt",
}

const jsontmpl = `{
	"sha256_asset_Decoder_pt"   : "%s",
	"sha256_asset_DVAE_full_pt"      : "%s",
	"sha256_asset_GPT_pt"       : "%s",
	"sha256_asset_spk_stat_pt"  : "%s",
	"sha256_asset_tokenizer_pt" : "%s",
	"sha256_asset_Vocos_pt"     : "%s"
}
`
