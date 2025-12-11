#!/bin/bash
# Terminal Color Configuration  
# Add this to your ~/.bashrc or source it: source ~/terminal-colors.sh

# Enable color support for ls and grep
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# More ls aliases with color
alias ll='ls -alF --color=auto'
alias la='ls -A --color=auto'
alias l='ls -CF --color=auto'

# Git color configuration
git config --global color.ui auto
git config --global color.status auto
git config --global color.branch auto
git config --global color.diff auto
git config --global color.interactive auto

# Specific git color settings
git config --global color.status.added "green bold"
git config --global color.status.changed "yellow bold"
git config --global color.status.untracked "red bold"
git config --global color.branch.current "yellow reverse"
git config --global color.branch.local "yellow"
git config --global color.branch.remote "green"

# Less with color support (for man pages, git log, etc)
export LESS='-R'
export LESS_TERMCAP_mb=$'\E[1;31m'     # begin bold
export LESS_TERMCAP_md=$'\E[1;36m'     # begin blink
export LESS_TERMCAP_me=$'\E[0m'        # reset bold/blink
export LESS_TERMCAP_so=$'\E[01;44;33m' # begin reverse video
export LESS_TERMCAP_se=$'\E[0m'        # reset reverse video
export LESS_TERMCAP_us=$'\E[1;32m'     # begin underline
export LESS_TERMCAP_ue=$'\E[0m'        # reset underline

# GCC colors for compiler output
export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# Colorized prompt with git branch and virtual environment
# Color codes (for use in PS1)
COLOR_RED='\[\033[0;31m\]'
COLOR_GREEN='\[\033[0;32m\]'
COLOR_YELLOW='\[\033[1;33m\]'
COLOR_BLUE='\[\033[0;34m\]'
COLOR_MAGENTA='\[\033[0;35m\]'
COLOR_CYAN='\[\033[0;36m\]'
COLOR_WHITE='\[\033[1;37m\]'
COLOR_GRAY='\[\033[0;90m\]'
COLOR_RESET='\[\033[0m\]'

# Function to get git branch
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

if [ -n "$ZSH_VERSION" ]; then
    setopt PROMPT_SUBST
    PROMPT='%F{cyan}${VIRTUAL_ENV:+($(basename $VIRTUAL_ENV)) }%f%F{green}%n@%m%f:%F{blue}$(git rev-parse --show-prefix 2>/dev/null || echo %1~)%f%F{yellow}$(parse_git_branch)%f%# '
elif [ -n "$BASH_VERSION" ]; then
    PS1='\[\033[0;36m\]${VIRTUAL_ENV:+($(basename $VIRTUAL_ENV)) }\[\033[0;32m\]\u@\h\[\033[0m\]:\[\033[0;34m\]$(git rev-parse --show-prefix 2>/dev/null || echo "\W")\[\033[1;33m\]$(parse_git_branch)\[\033[0m\]\$ '
fi

# Python/pip colored output
export PYTHONUNBUFFERED=1
export PIP_REQUIRE_VIRTUALENV=false

# Enable color in Python tracebacks (if rich is installed)
if command -v python3 &> /dev/null && python3 -c "import rich" 2>/dev/null; then
    export PYTHONBREAKPOINT=rich.traceback.install
fi

echo "✓ Terminal colors enabled!"
