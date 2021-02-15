################################################################################
#
#   Makefile for ARC
#
################################################################################

test test-unittests:
	nosetests --nocapture --nologcapture --all-modules --verbose --with-coverage --cover-inclusive --cover-package=arc --cover-erase --cover-html --exe --cover-html-dir=testing/coverage

install-all:
	bash devtools/install_all.sh

install-autotst:
	bash devtools/install_autotst.sh

install-gcn:
	bash devtools/install_gcn.sh

install-gcn-cpu:
	bash devtools/install_gcn_cpu.sh

install-kinbot:
	bash devtools/install_kinbot.sh
