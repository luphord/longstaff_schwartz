# -*- coding: utf-8 -*-

'''Command Line Interface (CLI) for longstaff_schwartz.'''
import sys
import click


@click.command()
def main(args=None):
    '''Command Line Interface (CLI) for longstaff_schwartz.'''
    click.echo('No CLI functionality yet')
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
