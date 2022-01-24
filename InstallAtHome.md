
## Use Python on your own computer

If you want, you can run these practicals locally on your own computer. The only issue is that it is more complicated for us to support any comnputing iussues you might have, which is why we suggest you use binder.

However, it it is not so hard to set things up and to run  it for yourself.

### Install Python

Whilst you *can* probably use any version of Python, we suggest you install [`Anaconda Python`](https://docs.anaconda.com/anaconda/install/). If you are very short on computer space, you might prefer the cut-down version [`Miniconda`](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html).

### make it faster if you want

The command that installs packages (libraries of software) for `Anaconda Python` is called `conda`. There is a *faster* alternative to this, called [`mamba`](https://github.com/mamba-org/mamba). We suggest you use this, after you have set up Python, type:
    
    conda install mamba -n base -c conda-forge
    
Now, instead of using `conda`, we will use [`mamba`](https://github.com/mamba-org/mamba) to manage packages. If you really want to revert to `conda`, then wherever you see `mamba` below, replace it with `conda`.

### Clone the repository

The code repository is on [github](https://github.com/UCL-EO/Workshop2022), so we need to pull a local copy. 

You can do this in several ways:

* Using `PyCharm` and `Anaconda Navigator` (if you installed Anaconda)
* Using `GitHub Desktop`
* Using `git` in a shell

####  Using `PyCharm` and `Anaconda Navigator`

We do not support this here, but you will find various online notes if you want to follow this path.


#### Using `GitHub Desktop`

If you are using MacOS or Windows, you can use the tool [`GitHub Desktop`](https://desktop.github.com) to manage your github repositories. 

Download and install the tool. Then run it.

Use the menu item `File -> Clone Repository` and enter `UCL-EO/Workshop2022` under the `GitHub.com` tab. So long as you haven't done this before, you should be able to hit the `Clone` buytton for the cloning to take place. Take note of where the repoisitory will be placed on your comnputer, e.g. `/Users/plewis/Documents/GitHub/Workshop2022`.

Any time we update the repository, you just need to click `Fetch origin` to update.

After this point, just follow the same instructions as for using `git` directly at the section `Setting up the environment`.

#### Using `git`

Make sure you have `git`. If you type:


    which git


and it does not print anything, then you probably haven't got `git` installed. You may be able to install it with:

    mamba install git

or alternatively, search on the web for an installation of `git` for your operating system.

Clone  the repository. First, choose a location where you want to put it, and `cd` there:

    mkdir -p ~/Documents/GitHub
    cd ~/Documents/GitHub

Clone:

    git clone git://github.com/UCL-EO/Workshop2022.git

This will now have set up the directory `Workshop2022`. If you type:

    ls geog0133

You should see:

    Install.md		data			postBuild
    LICENSE			environment.yml		python
    Makefile		figures			requirements.txt
    README.md		make.bat		docs
    notebooks


##### Setting up the environment

Assuming you have cloned the repository to `~/Documents/GitHub/Workshop2022`, now open a shell (Terminal) and type:

    cd ~/Documents/GitHub/Workshop2022

or replacing `~/Documents/GitHub/Workshop2022` with the location of your repository if it is somewhere else.

Then, set up the environment with:

    mamba env create  --force -n uclnceo -f environment.yml

This will take a few minutes, but will create the environment `geog0133` which contains all of the libaries you need for this course.

Now, activate it (N.B. you have to use `conda` here):

    conda activate uclnceo
  
Next run the post-build configuration script (sets up itens for Jupyter notebooks and installs the correct kernel for the notebooks -- `conda env:uclnceo`):

    ./postBuild 

Now start Jupyter:

    jupyter notebook

This may open a browser window for you, or might just instruct you to copy and paste a URL, e.g.:

    http://127.0.0.1:8888/?token=4afdc076ec49592ca1059d957f0bccbce86e17ab838f61e0


Make sure you have the Jupyter window running in the browser.

Navigate to `notebooks` and start any notebook -- files ending with `.ipynb` (click on it in the browser).

Now, run the cell `In [1]:` to test that the required codes load correctly.

If there is a problem, go back over the steps above. If you still can't solve the problems, try connecting by a different route, and/or and contact the [owner](mailto: p.lewis@ucl.ac.uk?subject=[Workshop2022 setup problem]), explaining *exactly* what you did and what the problem was.


