class Chatbot {
    // define a constructor
    constructor(){
        this.args = {
            // define different argument
            openButton: document.querySelector('.chatbox__button'),
            Chatbox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }
        // define a state if the chatbox is open or close
        this.state = false;
        this.messages = [];
    }
    // Javascript function to display the messaeges
    display() {
        // extract arguments
        const { openButton, Chatbox, sendButton } = this.args; 
    // add to event listener by calling event listener
        openButton.addEventListener('click', () => this.toggleState(Chatbox));
        // if user click open button then send a message 
        sendButton.addEventListener('click', () => this.onSendButton(Chatbox));
    
        const node = Chatbox.querySelector('input');
        // if hit a enter button
        node.addEventListener('keyup', ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(Chatbox);
            }
        });
    }

    // implement a toggle state
    toggleState(chatbox){
        this.state = !this.state;
        // show or hide the box
        if(this.state){
            chatbox.classList.add('chatbox--active')
        }else{
            chatbox.classList.remove('chatbox--active')
        }
    }

    // define onSendButton
    onSendButton(chatbox){
        // extract the text from user input
        let textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === ""){
            return;
        }

        // send the message to chatbot
        let msg1 = { name: "User", message: text1}
        // the message key 'message: text1' must be the same with the python code declaration in app.py
        this.messages.push(msg1);
        // push this object in messages array

        // fect the predict
        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({message: text1}),
            mode: 'cors',
            headers: {
                'Content-type': 'application/json'
            },
        })
        // extract back json
        .then(r => r.json())
        .then(r => {
            let msg2 = {name: "Amir", message: r.answer};
            // the message key 'message: r.answer' must be the same with the python code declaration in app.py
            this.messages.push(msg2);
            // push the message to array
            this.updateChatText(chatbox)
            // update the chat text
            textField.value = ''
        
        }).catch((eror) =>{
            console.error('Eror:', error);
            this.updateChatText(chatbox)
            textField.value =''
        });
    }

    // update chat text
    updateChatText(chatbox){
        let html = '';
        this.messages.slice().reverse().forEach(function(item, ){
            if(item.name ==="Amir"){
                html +='<div class="messages__item messages__item--visitor">' + item.message + '</div>'   
            }else{
                html +='<div class="messages__item messages__item--operator">' + item.message + '</div>'   
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbot();
chatbox.display();