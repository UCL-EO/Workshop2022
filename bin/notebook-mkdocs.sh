#!/bin/bash 

# shell to sort nb
# documentation
# run bin/notebook-run.sh


here=$(pwd)
base="$(cd $(dirname "$0") &&  cd .. &&  pwd -P && cd "$here")"
repo=$(echo $base | awk -F/ '{print $NF}')

cd $base
here=$base
echo "----> running $0 from $here"
echo "--> location: $base"
echo "--> repo: $repo"

# HOME may not be set on windows
if [ -z "$HOME" ] ; then
  cd ~
  HOME=$(pwd)
  echo "--> HOME $HOME"
  cd "$base"
fi

cd "$base"

echo "--> creating markdown files in notebooks"
# make markdown from the ipynb in notebooks_lab/???_*ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.allow_errors=True \
    --nbformat=4 --ExecutePreprocessor.store_widget_state=True --to markdown notebooks/*.ipynb

for i in notebooks/*md
do
  sed < $i > /tmp/$$ 's/.ipynb/.md/g'
  mv /tmp/$$ $i
done

# rst files to md
for i in docs/*.rst
do
  echo $i
  pandoc $i -o `echo $i|sed 's/.rst/.md/'`
done
# mathjax for rendering latex
# see https://squidfunk.github.io/mkdocs-material/reference/mathjax/#arithmatex
mkdir -p javascripts
cat << EOF > javascripts/config.js
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
EOF

mkdocs build -v
echo "----> done running $0 from $base"
echo "to upload, run:  mkdocs gh-deploy --force"
