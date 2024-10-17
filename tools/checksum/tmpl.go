package main

var files = [...]string{
	"asset/Decoder.safetensors",
	"asset/DVAE.safetensors",
	"asset/Embed.safetensors",
	"asset/Vocos.safetensors",

	"asset/gpt/config.json",
	"asset/gpt/model.safetensors",

	"asset/tokenizer/special_tokens_map.json",
	"asset/tokenizer/tokenizer_config.json",
	"asset/tokenizer/tokenizer.json",
}

const jsontmpl = `{
	"sha256_asset_Decoder_safetensors": "%s",
	"sha256_asset_DVAE_safetensors"   : "%s",
	"sha256_asset_Embed_safetensors"  : "%s",
	"sha256_asset_Vocos_safetensors"  : "%s",

	"sha256_asset_gpt_config_json"         : "%s",
	"sha256_asset_gpt_model_safetensors"   : "%s",

	"sha256_asset_tokenizer_special_tokens_map_json": "%s",
	"sha256_asset_tokenizer_tokenizer_config_json"  : "%s",
	"sha256_asset_tokenizer_tokenizer_json"         : "%s"
}
`
