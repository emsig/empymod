# Command line options.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = empymod
SOURCEDIR     = .
BUILDDIR      = _build

# Will also be triggered if "make" is provided without argument.
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

linkcheck-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b linkcheck $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
