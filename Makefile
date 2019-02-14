#dataset = pubmed
#method = gcn
#task = classification

dataset = Wikipedia
method = deepwalk
task = link_predict
ms = mle_w#random_search b_opt
ms_name = $(shell echo $(ms) | sed "s/ /_/g")
#log_file = logs/l_$(dataset)_$(method)_$(task).log
log_file = logs/l_$(dataset)_$(method)_$(task)_$(ms_name).log
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

check:
	ps -ef | grep "python3 -u src/main.py"

test:
	echo $(shell echo $(ms) | sed "s/ /_/g")

.PHONY: clean
clean:
	rm */.*swp*
