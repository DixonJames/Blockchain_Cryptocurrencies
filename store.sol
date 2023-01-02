// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract Store{
    // assighn the contract creator as the store manager
    // can add other users or othere shop addresses
    mapping(address => bool) public admins;
    // hash dict to items owned by store
    mapping(string => Item) public inventory;
    address  payable public Owner;
    address  public creator;

    constructor() {
        Owner = payable(msg.sender); 
        creator = msg.sender; 
        admins[msg.sender] = true;
 
    }

    // events to anaounce
    event daysLeft(string id, uint daysl);
    event newAdmin(address admin_addr);
    event removedAdmin(address admin_addr);
    event payIn(address sender, uint amount);
    event newItem(string id, uint amount);
    event removedItem(string id, uint amount);
    event itemCheckResult(string id, bool result);
    event itemTransfered(string id, address sender, address receiver);
    




    // access modifiers:
    modifier manager_access{
        require(msg.sender == creator);
        _;
    }
    modifier admin_access{
        // checks to see if the senders details are in the admin hashdict
        //require(admins[msg.sender] == true);

        _;
    }


    // functions for the owner to add/remove admins
    function addAdmin(address new_admin) public manager_access{
        admins[new_admin] = true;
        emit newAdmin(new_admin);
    } 
    function removeAdmin(address new_admin) public manager_access{
        admins[new_admin] = false;
        emit removedAdmin(new_admin);
    } 


    // item availability status
    enum i_availability{
        available, unavailable, no_stock
    }
    // when item reaches sell by date, what to do with item
    enum termination_procedure{
        recycle, expire
    }
    
    // item attributes
    struct Item{
        string id; // unique id
        string name; // human understandable name
        i_availability availability; // current item availbility
        address store; // store currently holding item
        uint value; // value of one item
        uint quantity; // quiantity of the item held
        uint creation_date; // creation date of the item
        uint time_fresh; // how many days the item will stay sellable for
        termination_procedure expirery_procedure; // termiantion procdure for iems end of life
    }

    // transfer eth to the store owner from public
    function transferIn(uint _amount) public payable{ 
        Owner.transfer(_amount);
        emit payIn(msg.sender, _amount);
    }


    // item creation and destruction
    function addItem(string memory  _id, string memory _name, uint _value, uint _quantity, uint days_fresh) public admin_access
    {   
        // if alreadu in inventry add stock
        if (inventory[_id].value != 0){
            
            Item memory current_item = inventory[_id];
            current_item.quantity += _quantity;
            current_item.availability = i_availability.available;

            inventory[_id] = current_item;
            emit newItem( _id,  current_item.quantity);
        }
        else{
            // if not already in invetory craete and add the item to hashdict
            uint time_fresh = 1 hours * 24 * days_fresh;
            inventory[_id] = Item(_id, _name, i_availability.available, address(this), _value, _quantity, block.timestamp, time_fresh, termination_procedure.recycle);
            emit newItem( _id,  _quantity);
        }
        
    }

    // removes the item stock from hahsdict. also changes availablity
    function removeItem(string memory _id, uint quantity) public admin_access{
        Item memory toRemove = inventory[_id];
        
        toRemove.quantity = toRemove.quantity - quantity;

        if (toRemove.quantity <= 0){
            toRemove.quantity = 0;
            toRemove.availability = i_availability.no_stock;
        }

        inventory[_id] = toRemove;
        emit removedItem(_id, quantity);
    }
    
    // check that item exists with enough stock
    function checkItem(string memory _id, uint quantity) public  admin_access returns (bool) {
        Item memory i_ref = inventory[_id];
        if (i_ref.store != address(this)){
            emit itemCheckResult(_id, false);
            return false;
        }
        if (i_ref.availability == i_availability.unavailable){
            emit itemCheckResult(_id, false);
            return false;
        }
        if (i_ref.quantity < quantity){
            emit itemCheckResult(_id, false);
            return false;
        }
        emit itemCheckResult(_id, true);
        return true;

    }

    // transfer the item to another store contract instance
    // other store needs to have added this stores instace address to its admins
    function transferItem(address store_address, string memory item_id, uint quantity) public admin_access{
        Item memory i_ref = inventory[item_id];

        Store transfer_store = Store(store_address);

        if (checkItem(item_id, quantity)){
            // remove from our stock
            removeItem(item_id, quantity);
            // add to other stores stock
            transfer_store.addItem(item_id, i_ref.name, i_ref.value, quantity, i_ref.time_fresh);
            emit itemTransfered(item_id, msg.sender, store_address);
        }

    }

    // finds the number of days left untill an item expires
    function itemdaysLeft(string memory item_id)  public admin_access returns (uint){
        uint time_left = (inventory[item_id].creation_date + inventory[item_id].time_fresh) - block.timestamp;
        if(time_left < 0){
            return 0;
        }
        uint val = time_left / 60 / 60 / 24;
        if(val == 0){
            val += 1;
        }
        emit daysLeft(item_id, val);
        return val;
    }

    // checks to see if the item can be termainted
    // if it can be termainted follows termaintion procedure accoring to the termaintion options
    function termianteItem(string memory item_id) public admin_access{
        if (itemdaysLeft(item_id) == 0){
            Item memory i_ref = inventory[item_id];

            if (i_ref.expirery_procedure == termination_procedure.recycle){
                i_ref.value = i_ref.value /2;
                i_ref.expirery_procedure = termination_procedure.expire;

            }
            if (i_ref.expirery_procedure == termination_procedure.expire){
                removeItem(item_id, i_ref.quantity);
            }
            inventory[item_id] = i_ref;

        }

    }
    
}