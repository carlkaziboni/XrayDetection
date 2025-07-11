import sqlite3
import click
from flask import g, current_app
import os

def get_db():
    """Get a database connection."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Close the database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize the database."""
    db = get_db()
    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

@click.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

sqlite3.register_converter("timestamp", lambda x: x if x is None else sqlite3.datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

def init_app(app):
    """Initialize the app with database functions."""
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

    # Register the database connection
    app.config['DATABASE'] = 'xraydetectionsite.sqlite'
    app.config['SQLITE_CONVERTERS'] = {'timestamp': sqlite3.PARSE_DECLTYPES}
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
