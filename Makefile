dataset = Flickr
method = deepwalk
task = link_predict#classification
ms = mle_k random_search b_opt
log_file = logs/l_$(dataset)_$(method)_$(task).log
log_pid = logs/pid_$(dataset)_$(method)_$(task)

run:
	python3 -u src/main.py $(dataset) $(method) $(task) $(ms)

server_run:
	nohup make run > $(log_file) 2>&1 & echo $$! > $(log_pid)

log:
	tail $(log_file) -n 20

pid:
	cat $(log_pid)

kill:
	kill -9 `cat $(log_pid)`

check_run:
	ps -ef | grep "python3 -u src/main.py"

.PHONY: clean
clean:
	rm */.*swp*
