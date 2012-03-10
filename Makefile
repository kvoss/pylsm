SRCS := plugin.py

install : ${SRCS}
	cp $? ~/.gimp-2.6/plug-ins

.PHONY : install all
