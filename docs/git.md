## git workflow
This document explains the github based workflow.

### git one-time settings
You will need to do these one-time settings.
- [hub](https://github.com/github/hub) is required to convert an issue to a pull request.
- [git alias](https://gist.github.com/MisterJames/dc53921d481adb96e0145f8896296dd2) makes your life easier.
    ```
    $ brew install hub
    $ cat << END >> ~/.gitconfig
    [alias]
        # one-line log
        l = log --pretty=format:"%C(yellow)%h\\ %ad%Cred%d\\ %Creset%s%Cblue\\ [%cn]" --decorate --date=short

        a = add
        ap = add -p
        c = commit --verbose
        ca = commit -a --verbose
        cm = commit -m
        cam = commit -a -m
        m = commit --amend --verbose

        d = diff
        ds = diff --stat
        dc = diff --cached
        dl = diff HEAD^ HEAD

        s = status -s
        co = checkout
        cob = checkout -b
        # list branches sorted by last modified
        b = "!git for-each-ref --sort='-authordate' --format='%(authordate)%09%(objectname:short)%09%(refname)' refs/heads | sed -e 's-refs/heads/--'"
        br = branch

        # list aliases
        la = "!git config -l | grep alias | cut -c 7-"
    END
    ```

### git workflow
- Find out current branch

    ```
    $ git branch -av
    ```
- Pull updates from github

    ```
    $ git pull --rebase
    ```
- Switch branch

    ```
    $ git co <branch_name>
    ```
- Create a branch and work in the branch

    ```
    $ git cob ggo-<issue number>
    ```
- Commit and create a pull request

    ```
    $ git add <changed files>
    $ git cm "commit message"
    $ git push origin HEAD
    ```

    After the above is done, you will find a new feature branch is created on github. If you were working on an issue and want to conver the issue to a pull request, do this:

    ```
    $ hub pull-request -i <issue number>
    ```
- Update feature branch from master

    Before you create a pull request, you may want to rebase the master. Then, do this:
    ```
    $ git co master
    $ git pull --rebase
    $ git co ggo-<issue number>
    $ git rebase master
    ```
