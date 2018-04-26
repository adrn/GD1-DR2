LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out brf
CHECK_RERUN =

NAME = gd1

all: ${NAME}.pdf

${NAME}.pdf: ${NAME}.tex *.bib
	${LATEX} ${NAME}
	bibtex ${NAME}
	${LATEX} ${NAME}
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."

clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})
	${RM} *.aux
	${RM} *Notes.bib
