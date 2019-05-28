

from flask import Flask, render_template, url_for, flash, redirect
from forms import PostReview


app = Flask(__name__)


app.config['SECRET_KEY'] = '7f706edd7849867a3ddb059a3c0c3bc1'


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/postreview", methods=['GET', 'POST'])
def postreview():
    form = PostReview()
    if form.validate_on_submit():
        if form.review.data == 'helpful':
            flash(f'Thanks {form.username.data}! for posting your review, this was really usefull..!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Thanks {form.username.data}! for posting, really appreciate if you can share more details..', 'danger')
            return redirect(url_for('home'))
    return render_template('postreview.html', title='PostReview', form=form)


if __name__ == '__main__':
    app.run(debug=True)