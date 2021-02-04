Artificial Intelligence DD2380

A1 - Minimax
===

# Objective
The objective of this assignment is to implement the Minimax search algorithm for the best next
 possible move in the KTH Fishing Derby game tree.

The solution should be efficient, with a time limit per cycle of 75e-3 seconds.

# Instructions
Program a solution in the file player.py in the provided space for doing so. It is possible to submit the solution in
different files if player.py imports correctly and without errors each of them.

# Installation
The code runs in Python 3.7 AMD64.

You should start with a clean virtual environment and install the
requirements for the code to run.

In UNIX, in the skeleton directory, run:

```
$ sudo pip install virtualenvwrapper
$ . /usr/local/bin/virtualenvwrapper.sh
$ mkvirtualenv -p /usr/bin/python3.7 fishingderby
(fishingderby) $ pip install -r requirements.txt
```

In Windows, in the skeleton directory, run:

```
$ pip install virtualenvwrapper-win
```

And then close and open a new terminal.

```
$ mkvirtualenv -p C:\Users\<YourWindowsUser>\AppData\Local\Programs\Python\Python37\bin\python.exe fishingderby
(fishingderby) $ pip install -r requirements_win.txt
```

In Mac OS X:
1. Install **python 3.7**

   https://www.python.org/downloads/mac-osx/

2. Install **virtualenv** and **virtualenvwrapper**

   * Install them with pip3.

   ```undefined
   $ sudo pip3 install virtualenv
   $ sudo pip3 install virtualenvwrapper
   ```

   * Search for the path of **virtualenvwrapper.sh**

   ```
   $ which virtualenvwrapper.sh
   ```

   For example, in my machine, the location of my virtualenvwrapper.sh is `/Library/Frameworks/Python.framework/Versions/3.7/bin/virtualenvwrapper.sh`

   * Modify **.bash_profile** file

     Open `/Users/YourUsername/.bash_profile`:

     ```
     $ open -e .bash_profile
     ```

     Append it with:

     ```
     export WORKON_HOME=$HOME/.virtualenvs
     export VIRTUALENVWRAPPER_SCRIPT=/Library/Frameworks/Python.framework/Versions/3.7/bin/virtualenvwrapper.sh
     export VIRTUALENVWRAPPER_PYTHON=/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
     export VIRTUALENVWRAPPER_VIRTUALENV=/Library/Frameworks/Python.framework/Versions/3.7/bin/virtualenv
     source /Library/Frameworks/Python.framework/Versions/3.7/bin/virtualenvwrapper.sh
     ```

     Finally, to make our modification work, type in:

     ```
     $ source .bash_profile
     ```

3. Error debug

   If you have an error like this:

   ```
   [root@r saas]# virtualenv --no-site-packages --python=python3 venv_saas
   usage: virtualenv [--version] [--with-traceback] [-v | -q] [--app-data APP_DATA] [--clear-app-data] [--discovery {builtin}] [-p py] [--creator {builtin,cpython3-posix,venv}] [--se
                     [--activators comma_sep_list] [--clear] [--system-site-packages] [--symlinks | --copies] [--download | --no-download] [--extra-search-dir d [d ...]] [--pip versi
                     [--no-setuptools] [--no-wheel] [--symlink-app-data] [--prompt prompt] [-h]
                     dest
   virtualenv: error: unrecognized arguments: --no-site-packages
   ```

   The problem may be the version of virtualenv. Uninstall the current virtualenv and re-install it with version 16.7.9.

   ```
   $ sudo pip3 uninstall virtualenv
   $ sudo pip3 install virtualenv==16.7.9
   ```

4. Start fishingderby

   ```
   $ . /Library/Frameworks/Python.framework/Versions/3.7/bin/virtualenvwrapper.sh
   $ mkvirtualenv -p /Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7 fishingderby
   ```

   **Note:** make sure to find the path of your python 3.7. You can use `$ which python3.7` to do so. For example, in my machine, the path of my python 3.7 is `/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7`.

   * Go to the skeleton directory and run:

   ```
   (fishingderby) $ pip3 install -r requirements.txt
   ```

# Graphical Interface
To visualize your agent at work and understand the rules of the game better, we added a graphical
interface. You can start with:

```
(fishingderby) $ python3 main.py settings.yml
```

To play yourself using the keyboard (left, right, up, down), change the variable "player_type" in "settings.yml" to the value "human".

Note that can change the scenario of the game! In order to do so change "observations_file" in settings.yml.