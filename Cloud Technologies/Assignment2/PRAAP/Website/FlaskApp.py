from flask import Flask, render_template, url_for, flash, redirect
from forms import PostReviewForm


app = Flask(__name__)

app.config['SECRET_KEY'] = '4ed6b7e2171cb2607487925fe0f320db'



@app.route("/", methods=['GET', 'POST'])
def postreview():
    form = PostReviewForm()
    if form.validate_on_submit():
        if form.review.data == 'helpful':
            flash(f'Thanks for posting your review, this was really usefull..!', 'success')
            return redirect(url_for('postreview', _anchor='review_form'))
        else:
            flash('Thanks for posting, really appreciate if you can share more details..', 'danger')
            return redirect(url_for('postreview', _anchor='review_form'))
    return render_template('index.html', form=form)


if __name__ == '__main__':
	app.run(debug=True)