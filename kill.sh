#!/usr/bin/env bash

name='main_g.py'
kill $(ps aux | grep $name | grep -v grep | awk '{print $2}')
