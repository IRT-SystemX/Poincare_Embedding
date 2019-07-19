# EM_Hyperbolic


Markdown itself doesn't have a mechanism for embedding a PDF. However, Markdown accepts raw HTML in its input and passes it through unaltered. So the question you want to ask is: How would you embed a PDF in HTML? In order words, what HTML would one use to have a browser display a PDF embedded in an HTML page? You would just include that HTML in your Markdown document.

You can find lots of suggestions in answers to the question: Recommended way to embed PDF in HTML?. For example, this answer provides a nice solution with a fallback for older browsers (all credit goes to Suneel Omrey):

<object data="http://yoursite.com/the.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://yoursite.com/the.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>