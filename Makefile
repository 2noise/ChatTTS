webui_api:
	streamlit run webui_api.py --browser.gatherUsageStats False

server:
	uvicorn api:app --host 0.0.0.0 --port 8080

client:
	python client.py

image_tag=24.06.06
image_name=jackiexiao/chat_tts_api_ui:${image_tag}
build_docker:
	docker build . -f Dockerfile -t ${image_name}
	docker tag  ${image_name} jackiexiao/chat_tts_api_ui:latest


docker_push:
	docker push ${image_name}
	docker puash jackiexiao/chat_tts_api_ui:latest

tencent_prefix=ccr.ccs.tencentyun.com/text-to-speech/chat_tts_api_ui
docker_push_tencent:
	docker tag ${image_name} ${tencent_prefix}:${image_tag}
	docker push ${tencent_prefix}:${image_tag}
	docker tag ${image_name} ${tencent_prefix}:latest
	docker push ${tencent_prefix}:latest

run_docker:
	docker run --name chat_tts \
		--gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		-p 8080:8080 -p 8501:8501 \
		${tencent_prefix}:latest

save_docker: # docker load -i chat-tts.tar.gz
	docker save ${image_name} | gzip > chat-tts.tar.gz
