webui_api:
	streamlit run webui_api.py --browser.gatherUsageStats False

server:
	uvicorn api:app --host 0.0.0.0 --port 8080

client:
	python client.py

image_name=jackiexiao/chat_tts_api_ui:24.06.06
build_docker:
	docker build . -f Dockerfile -t ${image_name}
	docker tag  ${image_name} jackiexiao/chat_tts_api_ui:latest

docker_push:
	docker push ${image_name}
	docker puash jackiexiao/chat_tts_api_ui:latest

run_docker:
	docker run --name chat_tts \
		--gpus all --ipc=host \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		-p 8080:8080 -p 8501:8501 \
		jackiexiao/chat_tts_api_ui
