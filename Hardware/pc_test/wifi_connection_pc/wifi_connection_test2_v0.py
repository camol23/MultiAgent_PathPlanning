#!/usr/bin/env python3

import ev3_dc as ev3

jukebox = ev3.Jukebox(protocol=ev3.WIFI)
antemn = jukebox.song(ev3.EU_ANTEMN)

antemn.start()