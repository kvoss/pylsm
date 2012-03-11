SRCS := pythonlsm.py

install :
	chmod +x ${SRCS}
	cp ${SRCS} ~/.gimp-2.6/plug-ins

.PHONY : install
