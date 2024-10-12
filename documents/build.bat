
pdflatex %1.tex %1.pdf

IF NOT EXIST .\build (
    MKDIR build
)

move *.aux .\build
move *.log .\build