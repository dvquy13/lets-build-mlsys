.PHONY:
.ONESHELL:


up:
	docker run -it -d -p 9439:8080 --name label-studio -v $$(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest

down:
	docker rm -f label-studio
