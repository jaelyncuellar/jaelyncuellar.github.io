// script.js 


document.addEventListener("DOMContentLoaded", () => {
    // EMAIL SO BOTS DONT TOUCH
    const user = "jluvcuellar"; 
    const domain = "gmail.com"; 
    const emailLink = document.getElementById("email-link"); 

    if (emailLink) { 
        emailLink.href=`mailto:${user}@${domain}`; 
    }

    // COLLAPSIBLE
    const buttons = document.querySelectorAll(".collapsible");
    buttons.forEach(button => {
        button.addEventListener("click", () => {
            const content = button.closest("section").querySelector(".content");

            content.classList.toggle("show");

            const expanded = content.classList.contains("show");
            button.textContent = expanded ? "−" : "+";
            button.setAttribute("aria-expanded", expanded);
        });
    });
    // var coll = document.getElementsByClassName("collapsible");
    // for (var i = 0; i < coll.length; i++) {
    //     coll[i].addEventListener("click", function() {
    //         this.classList.toggle("active");
    //         var content = this.nextElementSibling;
    //         content.classList.toggle("show");
    //     });
    // }

});


